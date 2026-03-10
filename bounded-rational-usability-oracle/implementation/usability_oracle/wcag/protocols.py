"""
usability_oracle.wcag.protocols — WCAG 2.2 conformance checking protocols.

Structural interfaces for parsing WCAG criteria, evaluating conformance,
and reporting results.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Protocol, Sequence, runtime_checkable

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from usability_oracle.core.protocols import AccessibilityTree
    from usability_oracle.wcag.types import (
        ConformanceLevel,
        SuccessCriterion,
        WCAGGuideline,
        WCAGResult,
        WCAGViolation,
    )


# ═══════════════════════════════════════════════════════════════════════════
# WCAGParser — load and resolve WCAG 2.2 criteria
# ═══════════════════════════════════════════════════════════════════════════

@runtime_checkable
class WCAGParser(Protocol):
    """Parse and resolve WCAG 2.2 success criteria from a data source.

    Implementations may load from bundled JSON, a remote W3C feed,
    or a custom criteria subset.
    """

    def load_criteria(self) -> Sequence[SuccessCriterion]:
        """Load all WCAG 2.2 success criteria.

        Returns
        -------
        Sequence[SuccessCriterion]
            Complete set of WCAG 2.2 success criteria.
        """
        ...

    def load_guidelines(self) -> Sequence[WCAGGuideline]:
        """Load all WCAG 2.2 guidelines with their criteria.

        Returns
        -------
        Sequence[WCAGGuideline]
            Guidelines grouped by principle.
        """
        ...

    def criteria_for_level(
        self, level: ConformanceLevel
    ) -> Sequence[SuccessCriterion]:
        """Return criteria at or below the given conformance level.

        Parameters
        ----------
        level : ConformanceLevel
            Maximum conformance level to include.
        """
        ...

    def criterion_by_id(self, sc_id: str) -> Optional[SuccessCriterion]:
        """Look up a single criterion by its dotted id (e.g. ``"1.4.3"``)."""
        ...


# ═══════════════════════════════════════════════════════════════════════════
# WCAGEvaluator — evaluate conformance of an accessibility tree
# ═══════════════════════════════════════════════════════════════════════════

@runtime_checkable
class WCAGEvaluator(Protocol):
    """Evaluate a UI (via its accessibility tree) against WCAG 2.2 criteria.

    This is the primary conformance-checking interface.  Implementations
    check individual success criteria (contrast ratio, target size,
    keyboard navigability, label presence, etc.) and aggregate results.
    """

    def evaluate(
        self,
        tree: AccessibilityTree,
        level: ConformanceLevel,
        *,
        criteria_ids: Optional[Sequence[str]] = None,
    ) -> WCAGResult:
        """Run conformance evaluation.

        Parameters
        ----------
        tree : AccessibilityTree
            Parsed accessibility tree of the UI under test.
        level : ConformanceLevel
            Target conformance level (A / AA / AAA).
        criteria_ids : Optional[Sequence[str]]
            If provided, only evaluate these specific criteria.

        Returns
        -------
        WCAGResult
            Aggregate conformance result with violations.
        """
        ...

    def check_criterion(
        self,
        tree: AccessibilityTree,
        criterion: SuccessCriterion,
    ) -> Sequence[WCAGViolation]:
        """Check a single success criterion.

        Parameters
        ----------
        tree : AccessibilityTree
            Parsed accessibility tree.
        criterion : SuccessCriterion
            The criterion to evaluate.

        Returns
        -------
        Sequence[WCAGViolation]
            Violations found (empty if the criterion passes).
        """
        ...


# ═══════════════════════════════════════════════════════════════════════════
# WCAGReporter — format WCAG results for output
# ═══════════════════════════════════════════════════════════════════════════

@runtime_checkable
class WCAGReporter(Protocol):
    """Format WCAG conformance results for various output channels."""

    def format_result(
        self,
        result: WCAGResult,
        *,
        format: str = "json",
    ) -> str:
        """Render a WCAGResult to a string.

        Parameters
        ----------
        result : WCAGResult
            The conformance evaluation result.
        format : str
            Output format (``"json"``, ``"sarif"``, ``"html"``, ``"markdown"``).

        Returns
        -------
        str
            Formatted report.
        """
        ...

    def summary(self, result: WCAGResult) -> str:
        """One-line summary suitable for CI/CD log output."""
        ...


__all__ = [
    "WCAGEvaluator",
    "WCAGParser",
    "WCAGReporter",
]
