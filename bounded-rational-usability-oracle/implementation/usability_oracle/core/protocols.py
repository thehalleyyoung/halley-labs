"""
usability_oracle.core.protocols — Structural interfaces (Protocol / ABC).

Every pluggable component in the oracle pipeline is defined here as a
:class:`typing.Protocol`.  Concrete implementations live in their respective
subpackages (``accessibility``, ``alignment``, ``cognitive``, ...).
"""

from __future__ import annotations

from abc import abstractmethod
from typing import (
    Any,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Sequence,
    TypeVar,
    runtime_checkable,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from usability_oracle.core.types import (
        ActionId,
        CostTuple,
        PolicyDistribution,
        StateId,
        Trajectory,
    )
    from usability_oracle.core.enums import (
        BottleneckType,
        EditOperationType,
        PipelineStage,
        RegressionVerdict,
        Severity,
    )

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)


# ═══════════════════════════════════════════════════════════════════════════
# Accessibility Tree & Parsing
# ═══════════════════════════════════════════════════════════════════════════

class AccessibilityNode(Protocol):
    """Minimal protocol for a single node in an accessibility tree.

    Implementations must provide at least node identity, role, name,
    and access to children.
    """

    @property
    def node_id(self) -> str:
        """Globally unique identifier for this node."""
        ...

    @property
    def role(self) -> str:
        """Accessibility role (e.g. 'button', 'link')."""
        ...

    @property
    def name(self) -> str:
        """Accessible name / label."""
        ...

    @property
    def children(self) -> Sequence[AccessibilityNode]:
        """Ordered list of child nodes."""
        ...

    @property
    def parent_id(self) -> Optional[str]:
        """Parent node id (None for root)."""
        ...

    @property
    def depth(self) -> int:
        """Depth in the tree (root = 0)."""
        ...


class AccessibilityTree(Protocol):
    """Minimal protocol for a parsed accessibility tree.

    An accessibility tree is a rooted, ordered tree of
    :class:`AccessibilityNode` instances.
    """

    @property
    def root(self) -> AccessibilityNode:
        """Root node of the tree."""
        ...

    @property
    def node_count(self) -> int:
        """Total number of nodes in the tree."""
        ...

    def find_by_id(self, node_id: str) -> Optional[AccessibilityNode]:
        """Look up a node by its unique id (None if not found)."""
        ...

    def find_by_role(self, role: str) -> Sequence[AccessibilityNode]:
        """Return all nodes with the given role."""
        ...

    def traverse_preorder(self) -> Sequence[AccessibilityNode]:
        """Return nodes in pre-order traversal."""
        ...

    @property
    def max_depth(self) -> int:
        """Maximum depth of the tree."""
        ...


@runtime_checkable
class Parser(Protocol):
    """Parse a raw UI source into an accessibility tree.

    Implementations may accept HTML strings, platform-native accessibility
    snapshots (JSON), or live browser connections.

    Parameters
    ----------
    source : Any
        Raw UI representation (type depends on implementation).

    Returns
    -------
    AccessibilityTree
        Parsed and normalised accessibility tree.

    Raises
    ------
    ParseError
        If the source cannot be parsed.
    InvalidAccessibilityTreeError
        If the parsed tree is structurally invalid.
    """

    def parse(self, source: Any) -> AccessibilityTree:
        """Parse *source* and return an accessibility tree."""
        ...


# ═══════════════════════════════════════════════════════════════════════════
# Alignment
# ═══════════════════════════════════════════════════════════════════════════

class AlignmentMapping(Protocol):
    """Result of aligning two accessibility trees."""

    @property
    def matched_pairs(self) -> Sequence[tuple[str, str]]:
        """Pairs of (old_node_id, new_node_id)."""
        ...

    @property
    def added_nodes(self) -> Sequence[str]:
        """Node ids present only in the *new* tree."""
        ...

    @property
    def removed_nodes(self) -> Sequence[str]:
        """Node ids present only in the *old* tree."""
        ...

    @property
    def edit_distance(self) -> float:
        """Weighted edit distance between the two trees."""
        ...

    @property
    def match_confidence(self) -> Dict[tuple[str, str], float]:
        """Per-pair confidence scores in [0, 1]."""
        ...


class AlignmentResult(Protocol):
    """Full alignment outcome including per-pass details."""

    @property
    def mapping(self) -> AlignmentMapping:
        """The alignment mapping."""
        ...

    @property
    def confidence(self) -> float:
        """Overall alignment confidence in [0, 1]."""
        ...

    @property
    def passes_used(self) -> Sequence[str]:
        """Which alignment passes contributed matches."""
        ...


@runtime_checkable
class Aligner(Protocol):
    """Align nodes across two accessibility trees.

    The aligner produces a mapping of matched nodes, plus lists of
    added and removed nodes, along with an overall edit distance.

    Parameters
    ----------
    tree_a : AccessibilityTree
        Baseline (old) accessibility tree.
    tree_b : AccessibilityTree
        Updated (new) accessibility tree.

    Returns
    -------
    AlignmentResult
        Mapping of matched, added, and removed nodes.

    Raises
    ------
    AlignmentError
        If alignment is not possible (incompatible trees, etc.).
    IncompatibleTreesError
        If the trees are structurally incompatible for alignment.
    """

    def align(
        self, tree_a: AccessibilityTree, tree_b: AccessibilityTree
    ) -> AlignmentResult:
        """Align *tree_a* (baseline) against *tree_b* (updated)."""
        ...


# ═══════════════════════════════════════════════════════════════════════════
# Cognitive Cost Model
# ═══════════════════════════════════════════════════════════════════════════

class CostContext(Protocol):
    """Contextual information passed to the cost model.

    Provides the spatial layout, number of siblings, task state, etc.
    that are needed to evaluate Fitts', Hick-Hyman, and visual-search laws.
    """

    @property
    def source_bbox(self) -> Any:
        """Bounding box of the source element."""
        ...

    @property
    def target_bbox(self) -> Any:
        """Bounding box of the target element."""
        ...

    @property
    def num_siblings(self) -> int:
        """Number of sibling elements (for Hick-Hyman)."""
        ...

    @property
    def visible_set_size(self) -> int:
        """Number of visible items in the current view (for visual search)."""
        ...

    @property
    def elapsed_since_encoding(self) -> float:
        """Seconds since last information encoding (for WM decay)."""
        ...

    @property
    def motor_channel(self) -> str:
        """Active motor channel for this operation."""
        ...


@runtime_checkable
class CostModel(Protocol):
    """Compute the cognitive cost tuple for a single UI operation.

    The cost model applies one or more cognitive laws (Fitts', Hick-Hyman,
    visual search, working-memory decay) and returns a composite
    :class:`CostTuple`.

    Parameters
    ----------
    operation : str | EditOperationType
        The type of UI operation being costed.
    context : CostContext
        Contextual information (layout, set sizes, timing).

    Returns
    -------
    CostTuple
        Composite (mu, sigma_sq, kappa, lambda_) cognitive cost.

    Raises
    ------
    CostModelError
        If the cost cannot be computed (missing context, invalid params).
    """

    def compute_cost(self, operation: Any, context: Any) -> CostTuple:
        """Compute the cognitive cost of *operation* in *context*."""
        ...


# ═══════════════════════════════════════════════════════════════════════════
# MDP / Policy
# ═══════════════════════════════════════════════════════════════════════════

class MDP(Protocol):
    """Minimal protocol for a Markov Decision Process.

    States and actions are identified by strings.  The MDP defines
    transitions, immediate costs, initial state, and goal states.
    """

    @property
    def states(self) -> Sequence[str]:
        """All states in the MDP."""
        ...

    @property
    def actions(self) -> Sequence[str]:
        """All actions in the MDP."""
        ...

    def transitions(self, state: str, action: str) -> Sequence[tuple[str, float]]:
        """Return list of (next_state, probability) pairs."""
        ...

    def cost(self, state: str, action: str) -> float:
        """Immediate cost of taking *action* in *state*."""
        ...

    @property
    def initial_state(self) -> str:
        """The starting state."""
        ...

    @property
    def goal_states(self) -> frozenset[str]:
        """Set of absorbing goal states."""
        ...

    @property
    def num_states(self) -> int:
        """Number of states."""
        ...

    @property
    def num_actions(self) -> int:
        """Number of actions."""
        ...


class Policy(Protocol):
    """A (possibly stochastic) policy over an MDP."""

    @property
    def distribution(self) -> PolicyDistribution:
        """Full action distribution at every state."""
        ...

    @property
    def beta(self) -> float:
        """Rationality parameter under which this policy was computed."""
        ...

    @property
    def expected_cost(self) -> float:
        """Expected total cost under this policy."""
        ...

    @property
    def free_energy(self) -> float:
        """Free energy F(pi) = E_pi[R] - (1/beta) D_KL(pi || p_0)."""
        ...

    @property
    def value_function(self) -> Dict[str, float]:
        """State value function V(s)."""
        ...


@runtime_checkable
class PolicyComputer(Protocol):
    """Compute the bounded-rational policy for a given MDP and beta.

    Uses the free-energy formulation to find the policy that minimises
    F(pi) = E_pi[cost] + (1/beta) * D_KL(pi || p_0).

    Parameters
    ----------
    mdp : MDP
        The usability MDP.
    beta : float
        Rationality parameter (higher = more rational).

    Returns
    -------
    Policy
        Optimal bounded-rational policy.

    Raises
    ------
    PolicyError
        On numerical issues or non-convergence.
    NumericalInstabilityError
        On softmax overflow / underflow.
    """

    def compute_policy(self, mdp: Any, beta: float) -> Policy:
        """Compute the optimal bounded-rational policy."""
        ...


# ═══════════════════════════════════════════════════════════════════════════
# Bottleneck Classification
# ═══════════════════════════════════════════════════════════════════════════

class TrajectoryStats(Protocol):
    """Aggregate statistics over a set of trajectories."""

    @property
    def mean_cost(self) -> float:
        """Mean total cost across trajectories."""
        ...

    @property
    def variance(self) -> float:
        """Variance of total costs."""
        ...

    @property
    def num_trajectories(self) -> int:
        """Number of sampled trajectories."""
        ...

    @property
    def percentiles(self) -> Dict[int, float]:
        """Cost percentiles (e.g. {50: 2.3, 95: 4.1})."""
        ...


class CostBreakdown(Protocol):
    """Per-cognitive-law decomposition of total cost."""

    def cost_for_law(self, law: str) -> float:
        """Cost attributed to a specific cognitive law."""
        ...

    @property
    def total(self) -> float:
        """Total cost across all laws."""
        ...

    @property
    def fractions(self) -> Dict[str, float]:
        """Fraction of total cost per law."""
        ...


class BottleneckResult(Protocol):
    """A single classified bottleneck."""

    @property
    def bottleneck_type(self) -> str:
        """Bottleneck category (e.g. 'motor_difficulty')."""
        ...

    @property
    def severity(self) -> str:
        """Severity level (e.g. 'high')."""
        ...

    @property
    def fraction_of_total(self) -> float:
        """Fraction of total cost attributable to this bottleneck."""
        ...

    @property
    def description(self) -> str:
        """Human-readable description of the bottleneck."""
        ...

    @property
    def affected_nodes(self) -> Sequence[str]:
        """Node ids most affected by this bottleneck."""
        ...


@runtime_checkable
class BottleneckClassifier(Protocol):
    """Classify cognitive bottlenecks from trajectory statistics.

    Examines the per-law cost breakdown and identifies which cognitive
    resource is most strained.

    Parameters
    ----------
    trajectory_stats : TrajectoryStats
        Aggregated trajectory statistics.
    cost_breakdown : CostBreakdown
        Per-law cost decomposition.

    Returns
    -------
    list[BottleneckResult]
        Identified bottlenecks, ordered by severity.

    Raises
    ------
    BottleneckError
        On classification failure.
    """

    def classify(
        self,
        trajectory_stats: Any,
        cost_breakdown: Any,
    ) -> List[Any]:
        """Classify bottlenecks from statistics and cost breakdown."""
        ...


# ═══════════════════════════════════════════════════════════════════════════
# Repair Synthesis
# ═══════════════════════════════════════════════════════════════════════════

class RepairConstraints(Protocol):
    """Constraints for synthesised repairs."""

    @property
    def max_edits(self) -> int:
        """Maximum number of tree edits allowed."""
        ...

    @property
    def preserve_semantics(self) -> bool:
        """Whether repairs must preserve task semantics."""
        ...

    @property
    def allowed_operations(self) -> Sequence[str]:
        """Which edit operations are permitted."""
        ...


class RepairCandidate(Protocol):
    """A proposed UI repair."""

    @property
    def edits(self) -> Sequence[Any]:
        """Ordered list of edit operations."""
        ...

    @property
    def predicted_cost_reduction(self) -> float:
        """Estimated cost reduction (seconds)."""
        ...

    @property
    def confidence(self) -> float:
        """Confidence in the predicted improvement [0, 1]."""
        ...

    @property
    def bottleneck_addressed(self) -> str:
        """Which bottleneck this repair targets."""
        ...


@runtime_checkable
class RepairSynthesizer(Protocol):
    """Synthesise candidate repairs for a classified bottleneck.

    Parameters
    ----------
    bottleneck : BottleneckResult
        The bottleneck to repair.
    constraints : RepairConstraints
        Structural and semantic constraints on acceptable repairs.

    Returns
    -------
    list[RepairCandidate]
        Candidate repairs, ordered by predicted cost reduction.

    Raises
    ------
    RepairError
        On synthesis timeout or infeasibility.
    SynthesisTimeoutError
        If the synthesis exceeds the time budget.
    InfeasibleRepairError
        If no repair satisfies the constraints.
    """

    def synthesize(
        self,
        bottleneck: Any,
        constraints: Any,
    ) -> List[Any]:
        """Synthesise repair candidates for *bottleneck*."""
        ...


# ═══════════════════════════════════════════════════════════════════════════
# Output Formatting
# ═══════════════════════════════════════════════════════════════════════════

@runtime_checkable
class OutputFormatter(Protocol):
    """Format an oracle result into a human- or machine-readable string.

    Parameters
    ----------
    result : Any
        The oracle analysis result to format.

    Returns
    -------
    str
        Formatted output (JSON, SARIF, HTML, or console text).
    """

    def format(self, result: Any) -> str:
        """Format *result* to string."""
        ...


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline Stage Executor
# ═══════════════════════════════════════════════════════════════════════════

class StageResult(Protocol):
    """Result of executing a single pipeline stage."""

    @property
    def stage(self) -> str:
        """Name of the stage."""
        ...

    @property
    def success(self) -> bool:
        """Whether the stage completed successfully."""
        ...

    @property
    def output(self) -> Any:
        """Stage output data."""
        ...

    @property
    def duration_seconds(self) -> float:
        """Wall-clock execution time."""
        ...

    @property
    def diagnostics(self) -> Sequence[str]:
        """Diagnostic messages produced during execution."""
        ...


@runtime_checkable
class PipelineStageExecutor(Protocol):
    """Execute a single stage of the oracle pipeline.

    Parameters
    ----------
    input : Any
        Stage-specific input (output of the previous stage).
    config : Any
        Stage configuration.

    Returns
    -------
    StageResult
        Outcome with output data and diagnostics.

    Raises
    ------
    StageError
        On unrecoverable stage failure.
    """

    def execute(self, input: Any, config: Any) -> StageResult:
        """Execute the pipeline stage."""
        ...


# ═══════════════════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════════════════

class ValidationIssue(Protocol):
    """A single validation finding."""

    @property
    def path(self) -> str:
        """Dot-separated path to the problematic field."""
        ...

    @property
    def message(self) -> str:
        """Description of the issue."""
        ...

    @property
    def severity(self) -> str:
        """Severity level."""
        ...


class ValidationResult(Protocol):
    """Result of a validation pass."""

    @property
    def is_valid(self) -> bool:
        """True if no errors were found (warnings are OK)."""
        ...

    @property
    def issues(self) -> Sequence[ValidationIssue]:
        """All issues found during validation."""
        ...

    @property
    def error_count(self) -> int:
        """Number of error-severity issues."""
        ...

    @property
    def warning_count(self) -> int:
        """Number of warning-severity issues."""
        ...


@runtime_checkable
class Validator(Protocol):
    """Validate data structures before pipeline processing.

    Parameters
    ----------
    data : Any
        The data to validate.

    Returns
    -------
    ValidationResult
        Validation outcome with any issues found.
    """

    def validate(self, data: Any) -> ValidationResult:
        """Validate *data* and return results."""
        ...


# ═══════════════════════════════════════════════════════════════════════════
# Serialisation
# ═══════════════════════════════════════════════════════════════════════════

Self = TypeVar("Self", bound="Serializable")


@runtime_checkable
class Serializable(Protocol):
    """Protocol for objects that support dict-based serialisation.

    Implementations must provide ``to_dict`` and a classmethod
    ``from_dict`` that round-trips faithfully:

        obj == Type.from_dict(obj.to_dict())
    """

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-compatible dictionary."""
        ...

    @classmethod
    def from_dict(cls: type[Self], d: Dict[str, Any]) -> Self:
        """Deserialise from a dictionary."""
        ...


# ═══════════════════════════════════════════════════════════════════════════
# Cache Provider
# ═══════════════════════════════════════════════════════════════════════════

@runtime_checkable
class CacheProvider(Protocol):
    """Simple key-value cache for memoising expensive computations.

    Implementations may use in-memory dicts, SQLite, Redis, or the
    filesystem.  Keys are strings; values are arbitrary serialisable
    objects.
    """

    def get(self, key: str) -> Optional[Any]:
        """Retrieve a cached value, or *None* on miss."""
        ...

    def set(self, key: str, value: Any) -> None:
        """Store a value in the cache."""
        ...

    def has(self, key: str) -> bool:
        """Check for key presence without deserialising the value."""
        ...

    def delete(self, key: str) -> None:
        """Remove a key from the cache (no-op if absent)."""
        ...

    def clear(self) -> None:
        """Flush the entire cache."""
        ...

    @property
    def size(self) -> int:
        """Number of entries in the cache."""
        ...


# ═══════════════════════════════════════════════════════════════════════════
# WCAGEvaluator — WCAG 2.2 conformance evaluation (core-level protocol)
# ═══════════════════════════════════════════════════════════════════════════

@runtime_checkable
class WCAGEvaluator(Protocol):
    """Evaluate a UI against WCAG 2.2 success criteria.

    Core-level protocol for WCAG conformance checking.  The full
    type-rich interface lives in :mod:`usability_oracle.wcag.protocols`;
    this protocol provides the minimal contract for pipeline integration.
    """

    def evaluate(
        self,
        tree: AccessibilityTree,
        level: str,
        *,
        criteria_ids: Optional[Sequence[str]] = None,
    ) -> Any:
        """Run WCAG conformance evaluation.

        Parameters
        ----------
        tree : AccessibilityTree
            Parsed accessibility tree of the UI under test.
        level : str
            Target conformance level (``"A"``, ``"AA"``, or ``"AAA"``).
        criteria_ids : Optional[Sequence[str]]
            If provided, only evaluate these specific criteria.

        Returns
        -------
        Any
            A ``WCAGResult`` (from :mod:`usability_oracle.wcag.types`).
        """
        ...

    def check_criterion(
        self,
        tree: AccessibilityTree,
        criterion_id: str,
    ) -> Sequence[Any]:
        """Check a single WCAG success criterion by its id.

        Parameters
        ----------
        tree : AccessibilityTree
            Parsed accessibility tree.
        criterion_id : str
            Dotted criterion id (e.g. ``"1.4.3"``).

        Returns
        -------
        Sequence[Any]
            Violations found (empty if the criterion passes).
        """
        ...


# ═══════════════════════════════════════════════════════════════════════════
# GomsPredictor — GOMS/KLM task-time prediction (core-level protocol)
# ═══════════════════════════════════════════════════════════════════════════

@runtime_checkable
class GomsPredictor(Protocol):
    """Predict task completion time using GOMS / KLM cognitive architecture.

    Core-level protocol for GOMS-based task-time estimation.
    Full type-rich interface in :mod:`usability_oracle.goms.protocols`.
    """

    def predict_task_time(
        self,
        tree: AccessibilityTree,
        task_description: str,
    ) -> float:
        """Predict task completion time in seconds.

        Parameters
        ----------
        tree : AccessibilityTree
            Parsed accessibility tree.
        task_description : str
            Natural-language task description.

        Returns
        -------
        float
            Predicted task completion time in seconds.
        """
        ...

    def compare_task_times(
        self,
        tree_old: AccessibilityTree,
        tree_new: AccessibilityTree,
        task_description: str,
    ) -> Dict[str, Any]:
        """Compare predicted task times between two UI versions.

        Parameters
        ----------
        tree_old : AccessibilityTree
            Accessibility tree of the old UI version.
        tree_new : AccessibilityTree
            Accessibility tree of the new UI version.
        task_description : str
            Task to compare.

        Returns
        -------
        Dict[str, Any]
            Comparison result with keys ``"time_old_s"``, ``"time_new_s"``,
            ``"delta_s"``, ``"regression"`` (bool).
        """
        ...


# ═══════════════════════════════════════════════════════════════════════════
# ChannelAnalyzer — MRT channel capacity analysis (core-level protocol)
# ═══════════════════════════════════════════════════════════════════════════

@runtime_checkable
class ChannelAnalyzer(Protocol):
    """Analyse Multiple Resource Theory channel utilisation for a UI task.

    Core-level protocol for MRT-based channel analysis.
    Full type-rich interface in :mod:`usability_oracle.channel.protocols`.
    """

    def analyze_channel_load(
        self,
        tree: AccessibilityTree,
        task_description: str,
    ) -> Dict[str, float]:
        """Compute per-channel load for a task on a UI.

        Parameters
        ----------
        tree : AccessibilityTree
            Parsed accessibility tree.
        task_description : str
            Task description.

        Returns
        -------
        Dict[str, float]
            Mapping from channel name to utilisation fraction [0, 1].
        """
        ...

    def identify_bottleneck_channels(
        self,
        tree: AccessibilityTree,
        task_description: str,
        *,
        threshold: float = 0.8,
    ) -> Sequence[str]:
        """Identify resource channels exceeding a utilisation threshold.

        Parameters
        ----------
        tree : AccessibilityTree
            Parsed accessibility tree.
        task_description : str
            Task description.
        threshold : float
            Utilisation threshold above which a channel is a bottleneck.

        Returns
        -------
        Sequence[str]
            Names of bottleneck channels.
        """
        ...


__all__ = [
    "AccessibilityNode",
    "AccessibilityTree",
    "AlignmentMapping",
    "AlignmentResult",
    "Aligner",
    "BottleneckClassifier",
    "BottleneckResult",
    "CacheProvider",
    "ChannelAnalyzer",
    "CostBreakdown",
    "CostContext",
    "CostModel",
    "GomsPredictor",
    "MDP",
    "OutputFormatter",
    "Parser",
    "PipelineStageExecutor",
    "Policy",
    "PolicyComputer",
    "RepairCandidate",
    "RepairConstraints",
    "RepairSynthesizer",
    "Serializable",
    "StageResult",
    "TrajectoryStats",
    "ValidationIssue",
    "ValidationResult",
    "Validator",
    "WCAGEvaluator",
]
