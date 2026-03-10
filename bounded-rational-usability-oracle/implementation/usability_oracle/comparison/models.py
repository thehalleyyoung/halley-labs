"""
usability_oracle.comparison.models — Data structures for paired usability comparison.

Defines :class:`ComparisonResult`, :class:`BottleneckChange`,
:class:`RegressionReport`, :class:`ComparisonContext`, and supporting
alignment/partition types needed by the comparison pipeline.

The comparison module builds a union MDP from two UI versions, computes
bounded-rational policies at matching rationality parameters, and tests
for statistically significant cost regressions.

References
----------
- Todorov, E. (2007). Linearly-solvable Markov decision processes. *NIPS*.
- Ortega, P. A. & Braun, D. A. (2013). Thermodynamics as a theory of
  decision-making with information-processing costs. *Proc. R. Soc. A*, 469.
- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences*.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

from usability_oracle.core.enums import BottleneckType, RegressionVerdict, Severity
from usability_oracle.cognitive.models import CostElement
from usability_oracle.mdp.models import MDP
from usability_oracle.taskspec.models import TaskSpec


# ---------------------------------------------------------------------------
# Alignment types (alignment/ module is not yet populated)
# ---------------------------------------------------------------------------

@dataclass
class StateMapping:
    """A single mapping between a state in MDP-A and a state in MDP-B.

    Attributes
    ----------
    state_a : str
        State ID in the *before* MDP.
    state_b : str
        State ID in the *after* MDP.
    similarity : float
        Alignment confidence in [0, 1].
    mapping_type : str
        One of ``"exact"``, ``"structural"``, ``"semantic"``, ``"heuristic"``.
    """

    state_a: str
    state_b: str
    similarity: float = 1.0
    mapping_type: str = "exact"


@dataclass
class AlignmentResult:
    """Result of aligning two accessibility trees / MDPs.

    The alignment establishes a partial bijection between states of the
    *before* and *after* MDPs so that costs can be compared at
    corresponding interaction points.

    Attributes
    ----------
    mappings : list[StateMapping]
        Individual state-to-state correspondences.
    unmapped_a : list[str]
        State IDs in MDP-A with no counterpart in MDP-B (deletions).
    unmapped_b : list[str]
        State IDs in MDP-B with no counterpart in MDP-A (additions).
    overall_similarity : float
        Global similarity score in [0, 1].
    metadata : dict
        Auxiliary alignment metadata (algorithm, runtime, etc.).
    """

    mappings: list[StateMapping] = field(default_factory=list)
    unmapped_a: list[str] = field(default_factory=list)
    unmapped_b: list[str] = field(default_factory=list)
    overall_similarity: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    # Convenience helpers --------------------------------------------------

    def get_mapping_dict(self) -> dict[str, str]:
        """Return ``{state_a: state_b}`` lookup."""
        return {m.state_a: m.state_b for m in self.mappings}

    def get_reverse_mapping(self) -> dict[str, str]:
        """Return ``{state_b: state_a}`` lookup."""
        return {m.state_b: m.state_a for m in self.mappings}

    @property
    def n_mapped(self) -> int:
        return len(self.mappings)

    @property
    def n_unmapped(self) -> int:
        return len(self.unmapped_a) + len(self.unmapped_b)


# ---------------------------------------------------------------------------
# Partition type (bisimulation/ module is not yet populated)
# ---------------------------------------------------------------------------

@dataclass
class PartitionBlock:
    """A single block in a state-space partition.

    Attributes
    ----------
    block_id : str
        Unique identifier for this block.
    state_ids : list[str]
        States belonging to this block.
    representative : str
        A canonical representative state for the block.
    """

    block_id: str
    state_ids: list[str] = field(default_factory=list)
    representative: str = ""


@dataclass
class Partition:
    """State-space partition for bisimulation quotient.

    Attributes
    ----------
    blocks : list[PartitionBlock]
        Disjoint blocks covering the state space.
    state_to_block : dict[str, str]
        Maps each state ID to its block ID.
    """

    blocks: list[PartitionBlock] = field(default_factory=list)
    state_to_block: dict[str, str] = field(default_factory=dict)

    @property
    def n_blocks(self) -> int:
        return len(self.blocks)

    def get_block(self, state_id: str) -> Optional[PartitionBlock]:
        """Return the block containing *state_id*, or ``None``."""
        bid = self.state_to_block.get(state_id)
        if bid is None:
            return None
        for b in self.blocks:
            if b.block_id == bid:
                return b
        return None


# ---------------------------------------------------------------------------
# Bottleneck change
# ---------------------------------------------------------------------------

class ChangeDirection(str, Enum):
    """Direction of change for a bottleneck between UI versions."""

    NEW = "new"
    RESOLVED = "resolved"
    WORSENED = "worsened"
    IMPROVED = "improved"


@dataclass
class BottleneckChange:
    """Describes how a cognitive bottleneck changed between versions.

    Attributes
    ----------
    bottleneck_type : BottleneckType
        Category of the bottleneck (perceptual overload, choice paralysis, …).
    state_id : str
        State where the bottleneck is located.
    before_severity : float
        Severity score in the *before* version (0 = absent).
    after_severity : float
        Severity score in the *after* version (0 = absent).
    direction : ChangeDirection
        Summary direction of the change.
    description : str
        Human-readable explanation.
    """

    bottleneck_type: BottleneckType
    state_id: str = ""
    before_severity: float = 0.0
    after_severity: float = 0.0
    direction: ChangeDirection = ChangeDirection.NEW
    description: str = ""

    @classmethod
    def classify_direction(
        cls, before: float, after: float, threshold: float = 0.01
    ) -> ChangeDirection:
        """Determine the change direction from severity values.

        Parameters
        ----------
        before : float
            Severity in the *before* version.
        after : float
            Severity in the *after* version.
        threshold : float
            Minimum absolute change to count as non-zero.

        Returns
        -------
        ChangeDirection
        """
        if before < threshold and after >= threshold:
            return ChangeDirection.NEW
        if before >= threshold and after < threshold:
            return ChangeDirection.RESOLVED
        delta = after - before
        if delta > threshold:
            return ChangeDirection.WORSENED
        if delta < -threshold:
            return ChangeDirection.IMPROVED
        return ChangeDirection.IMPROVED  # negligible change


# ---------------------------------------------------------------------------
# ComparisonResult
# ---------------------------------------------------------------------------

@dataclass
class ComparisonResult:
    """Full result of a paired usability comparison.

    Captures the regression verdict, statistical evidence, cost
    decomposition, bottleneck changes, and parameter-sensitivity metadata.

    Attributes
    ----------
    verdict : RegressionVerdict
        Overall verdict (``REGRESSION``, ``IMPROVEMENT``, ``NO_CHANGE``,
        ``INCONCLUSIVE``).
    confidence : float
        1 − α, the statistical confidence level.
    p_value : float
        p-value from the hypothesis test.
    cost_before : CostElement
        Aggregate cost for the *before* version.
    cost_after : CostElement
        Aggregate cost for the *after* version.
    delta_cost : CostElement
        ``cost_after − cost_before`` (positive ⇒ regression).
    effect_size : float
        Cohen's *d* effect size.
    bottleneck_changes : list[BottleneckChange]
        Per-bottleneck change descriptions.
    parameter_sensitivity : dict[str, float]
        Sensitivity of the verdict to each model parameter.
    is_parameter_free : bool
        ``True`` if the verdict is unanimous across all β in the tested range.
    description : str
        Natural-language summary of the result.

    References
    ----------
    Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences*.
    """

    verdict: RegressionVerdict = RegressionVerdict.INCONCLUSIVE
    confidence: float = 0.95
    p_value: float = 1.0
    cost_before: CostElement = field(
        default_factory=lambda: CostElement(
            mean_time=0.0, variance=0.0, channel="aggregate", law="composite"
        )
    )
    cost_after: CostElement = field(
        default_factory=lambda: CostElement(
            mean_time=0.0, variance=0.0, channel="aggregate", law="composite"
        )
    )
    delta_cost: CostElement = field(
        default_factory=lambda: CostElement(
            mean_time=0.0, variance=0.0, channel="aggregate", law="composite"
        )
    )
    effect_size: float = 0.0
    bottleneck_changes: list[BottleneckChange] = field(default_factory=list)
    parameter_sensitivity: dict[str, float] = field(default_factory=dict)
    is_parameter_free: bool = False
    description: str = ""

    # Derived properties ---------------------------------------------------

    @property
    def is_regression(self) -> bool:
        return self.verdict == RegressionVerdict.REGRESSION

    @property
    def is_improvement(self) -> bool:
        return self.verdict == RegressionVerdict.IMPROVEMENT

    @property
    def effect_magnitude(self) -> str:
        """Classify effect size per Cohen's (1988) conventions."""
        d = abs(self.effect_size)
        if d < 0.2:
            return "negligible"
        if d < 0.5:
            return "small"
        if d < 0.8:
            return "medium"
        return "large"


# ---------------------------------------------------------------------------
# RegressionReport
# ---------------------------------------------------------------------------

@dataclass
class RegressionReport:
    """Complete regression report aggregating per-task results.

    Attributes
    ----------
    comparison_result : ComparisonResult
        Overall (aggregate) comparison result.
    task_results : dict[str, ComparisonResult]
        Per-task comparison results keyed by task ID.
    overall_verdict : RegressionVerdict
        Conservative aggregate verdict.
    recommendations : list[str]
        Actionable recommendations for UI improvement.
    metadata : dict
        Report metadata (timestamp, tool version, etc.).
    """

    comparison_result: ComparisonResult = field(default_factory=ComparisonResult)
    task_results: dict[str, ComparisonResult] = field(default_factory=dict)
    overall_verdict: RegressionVerdict = RegressionVerdict.INCONCLUSIVE
    recommendations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_tasks(self) -> int:
        return len(self.task_results)

    @property
    def regression_tasks(self) -> list[str]:
        """Task IDs that exhibit a regression."""
        return [
            tid for tid, r in self.task_results.items()
            if r.verdict == RegressionVerdict.REGRESSION
        ]

    @property
    def improved_tasks(self) -> list[str]:
        """Task IDs that exhibit an improvement."""
        return [
            tid for tid, r in self.task_results.items()
            if r.verdict == RegressionVerdict.IMPROVEMENT
        ]


# ---------------------------------------------------------------------------
# ComparisonContext
# ---------------------------------------------------------------------------

@dataclass
class ComparisonContext:
    """Contextual inputs for a comparison run.

    Bundles the two MDPs, their alignment, the task specification,
    and the comparison configuration so that downstream functions
    receive a single coherent context object.

    Attributes
    ----------
    mdp_before : MDP
        MDP constructed from the *before* UI version.
    mdp_after : MDP
        MDP constructed from the *after* UI version.
    alignment : AlignmentResult
        State-level alignment between the two MDPs.
    task_spec : TaskSpec
        The task specification being evaluated.
    config : dict
        Comparison configuration parameters.
    """

    mdp_before: MDP = field(default_factory=MDP)
    mdp_after: MDP = field(default_factory=MDP)
    alignment: AlignmentResult = field(default_factory=AlignmentResult)
    task_spec: TaskSpec = field(default_factory=TaskSpec)
    config: dict[str, Any] = field(default_factory=dict)
