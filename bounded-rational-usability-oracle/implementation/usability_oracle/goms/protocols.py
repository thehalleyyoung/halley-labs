"""
usability_oracle.goms.protocols — GOMS / KLM cognitive architecture protocols.

Structural interfaces for GOMS analysis, KLM task-time prediction,
and method-level optimisation.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Protocol, Sequence, runtime_checkable

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from usability_oracle.core.protocols import AccessibilityTree
    from usability_oracle.goms.types import (
        GomsGoal,
        GomsMethod,
        GomsModel,
        GomsTrace,
        KLMSequence,
    )


# ═══════════════════════════════════════════════════════════════════════════
# GomsAnalyzer — build and analyse GOMS models from UI trees
# ═══════════════════════════════════════════════════════════════════════════

@runtime_checkable
class GomsAnalyzer(Protocol):
    """Build GOMS models from accessibility trees and task specifications.

    Constructs the goal–method–operator hierarchy for a given task,
    then computes predicted execution times via selection-rule resolution.
    """

    def build_model(
        self,
        tree: AccessibilityTree,
        task_description: str,
        *,
        top_level_goal: Optional[str] = None,
    ) -> GomsModel:
        """Construct a GOMS model for a task on the given UI.

        Parameters
        ----------
        tree : AccessibilityTree
            Parsed UI accessibility tree.
        task_description : str
            Natural-language or structured task description.
        top_level_goal : Optional[str]
            Name for the top-level goal (auto-generated if None).

        Returns
        -------
        GomsModel
            Complete GOMS model.
        """
        ...

    def trace(
        self,
        model: GomsModel,
        *,
        selection_policy: Optional[Mapping[str, str]] = None,
    ) -> GomsTrace:
        """Execute a GOMS model trace using selection rules.

        Parameters
        ----------
        model : GomsModel
            The GOMS model to trace.
        selection_policy : Optional[Mapping[str, str]]
            Override selection rules: goal_id → preferred method_id.

        Returns
        -------
        GomsTrace
            Execution trace with timing predictions.
        """
        ...

    def compare_traces(
        self,
        trace_old: GomsTrace,
        trace_new: GomsTrace,
    ) -> Mapping[str, Any]:
        """Compare two GOMS traces (e.g. before/after UI change).

        Returns a mapping with keys like ``"time_delta_s"``,
        ``"operator_count_delta"``, ``"regression"`` (bool).
        """
        ...


# ═══════════════════════════════════════════════════════════════════════════
# KLMPredictor — flat keystroke-level model task time prediction
# ═══════════════════════════════════════════════════════════════════════════

@runtime_checkable
class KLMPredictor(Protocol):
    """Predict task completion time using the Keystroke-Level Model.

    Generates a flat operator sequence (no goal hierarchy) with
    M-operator placement heuristics applied.
    """

    def predict(
        self,
        tree: AccessibilityTree,
        task_description: str,
    ) -> KLMSequence:
        """Generate a KLM operator sequence and predict task time.

        Parameters
        ----------
        tree : AccessibilityTree
            Parsed accessibility tree.
        task_description : str
            Task to perform.

        Returns
        -------
        KLMSequence
            Operator sequence with total time prediction.
        """
        ...

    def apply_mental_prep_heuristics(
        self,
        sequence: KLMSequence,
    ) -> KLMSequence:
        """Apply M-operator placement heuristics to a raw sequence.

        Uses the standard KLM rules (Raskin 2000 / Card et al. 1980)
        for inserting mental preparation operators.

        Parameters
        ----------
        sequence : KLMSequence
            Raw sequence without M operators.

        Returns
        -------
        KLMSequence
            Sequence with M operators placed.
        """
        ...


# ═══════════════════════════════════════════════════════════════════════════
# GomsOptimizer — optimise method selection or UI layout for GOMS
# ═══════════════════════════════════════════════════════════════════════════

@runtime_checkable
class GomsOptimizer(Protocol):
    """Suggest UI or method-selection changes that improve GOMS metrics.

    Given a GOMS model and trace, identify bottleneck operators and
    propose modifications that reduce predicted task time.
    """

    def identify_bottleneck_operators(
        self,
        trace: GomsTrace,
        *,
        threshold_fraction: float = 0.2,
    ) -> Sequence[str]:
        """Return operator descriptions consuming > threshold of total time.

        Parameters
        ----------
        trace : GomsTrace
            Execution trace to analyse.
        threshold_fraction : float
            Fraction of total time above which an operator is a bottleneck.

        Returns
        -------
        Sequence[str]
            Descriptions of bottleneck operators.
        """
        ...

    def suggest_improvements(
        self,
        model: GomsModel,
        trace: GomsTrace,
    ) -> Sequence[Mapping[str, Any]]:
        """Suggest improvements to the UI that reduce task time.

        Each suggestion is a mapping with keys: ``"description"``,
        ``"expected_time_saving_s"``, ``"operator_affected"``,
        ``"confidence"``.
        """
        ...


__all__ = [
    "GomsAnalyzer",
    "GomsOptimizer",
    "KLMPredictor",
]
