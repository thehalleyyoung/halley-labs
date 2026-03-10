"""
usability_oracle.bottleneck.repair_map — Repair strategy mapper.

Maps detected bottleneck types to concrete repair strategies.  Each
bottleneck type has a prioritised list of applicable repairs drawn from
the HCI literature and UI design best practices.

The mapper also contextualises repairs based on the bottleneck's severity
and the specific evidence (e.g., "increase target size to 44px" when the
current size is 18px).

Repair strategies are organised by bottleneck type:

  - **Perceptual overload** → reduce clutter, improve grouping, increase saliency
  - **Choice paralysis** → reduce options, progressive disclosure, defaults
  - **Motor difficulty** → larger targets, shorter distances, keyboard shortcuts
  - **Memory decay** → fewer steps, persistent cues, summaries
  - **Cross-channel interference** → separate channels, serialise interactions

References
----------
- Nielsen, J. (1994). *Usability Engineering* (heuristic evaluation).
- Shneiderman, B. & Plaisant, C. (2010). *Designing the User Interface*.
- WCAG 2.1 Success Criteria.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from usability_oracle.bottleneck.models import BottleneckResult
from usability_oracle.core.enums import BottleneckType, Severity


# ---------------------------------------------------------------------------
# Repair strategy registry
# ---------------------------------------------------------------------------

_PERCEPTUAL_REPAIRS: list[str] = [
    "Reduce the number of visible interactive elements (target: ≤ 7 primary actions)",
    "Group related elements using proximity, whitespace, or visual containers",
    "Increase visual saliency of primary actions (contrast ratio ≥ 4.5:1)",
    "Remove decorative elements that compete for visual attention",
    "Use progressive disclosure to hide secondary options behind expand/collapse",
    "Apply consistent visual hierarchy (size, colour, weight) to differentiate levels",
    "Reduce colour palette to ≤ 5 distinct hues for interactive elements",
    "Add clear section headings to chunk content into scannable groups",
    "Increase whitespace between distinct element groups by ≥ 16px",
    "Ensure the primary call-to-action is visually dominant (≥ 2× size of secondary)",
]

_CHOICE_REPAIRS: list[str] = [
    "Reduce visible options to 3–5 primary choices per decision point",
    "Add a recommended/default option with visual distinction",
    "Implement progressive disclosure (show top choices, 'More options…' link)",
    "Group related options into categories with clear labels",
    "Provide smart defaults based on user context or common patterns",
    "Add comparison tables for complex multi-attribute decisions",
    "Use wizard/step-by-step flows to decompose complex decisions",
    "Remove rarely-used options or move them to advanced settings",
    "Add search/filter for option sets larger than 10 items",
    "Provide 'undo' to reduce the perceived cost of choosing wrong",
]

_MOTOR_REPAIRS: list[str] = [
    "Increase target size to at least 44×44 CSS pixels (WCAG 2.5.5 AAA)",
    "Reduce distance between sequential interaction targets (≤ 200px ideal)",
    "Add keyboard shortcuts for frequent actions (Ctrl+S, Enter to submit)",
    "Place primary actions in Fitts' law 'hot zones' (corners, edges, centre)",
    "Add generous click/touch padding around interactive elements",
    "Use sticky/fixed positioning for frequently-used controls",
    "Provide drag handles with ≥ 32px grab area for drag interactions",
    "Implement snap-to-target for precision alignment tasks",
    "Replace distant click targets with contextual menus (right-click / long-press)",
    "Reduce the number of required precise pointing gestures per task",
]

_MEMORY_REPAIRS: list[str] = [
    "Reduce the number of steps in multi-page workflows to ≤ 3",
    "Show previously entered information on subsequent pages (persist context)",
    "Add breadcrumb navigation showing task progress",
    "Provide a summary/review step before final submission",
    "Display inline validation rather than deferred error messages",
    "Auto-save form state to prevent loss on navigation",
    "Chunk information into groups of ≤ 4 items (Cowan's limit)",
    "Add persistent visual indicators for current mode/state",
    "Replace recall with recognition: show options rather than requiring memory",
    "Provide contextual help/tooltips for fields requiring remembered information",
]

_INTERFERENCE_REPAIRS: list[str] = [
    "Serialise concurrent interaction requirements (one input mode at a time)",
    "Separate visual and manual tasks temporally (show → click, not simultaneously)",
    "Offload visual channel with auditory feedback (success sounds, error tones)",
    "Avoid requiring simultaneous spatial reasoning and manual input",
    "Provide single-modality alternatives for multimodal interactions",
    "Use modal dialogs to focus attention on one task at a time",
    "Reduce animation during active input tasks (visual channel competition)",
    "Provide haptic feedback to reduce visual monitoring of motor actions",
    "Separate read-then-act patterns with clear visual transitions",
    "Allow task switching rather than requiring concurrent processing",
]

REPAIR_STRATEGIES: dict[BottleneckType, list[str]] = {
    BottleneckType.PERCEPTUAL_OVERLOAD: _PERCEPTUAL_REPAIRS,
    BottleneckType.CHOICE_PARALYSIS: _CHOICE_REPAIRS,
    BottleneckType.MOTOR_DIFFICULTY: _MOTOR_REPAIRS,
    BottleneckType.MEMORY_DECAY: _MEMORY_REPAIRS,
    BottleneckType.CROSS_CHANNEL_INTERFERENCE: _INTERFERENCE_REPAIRS,
}


# ---------------------------------------------------------------------------
# RepairMapper
# ---------------------------------------------------------------------------

@dataclass
class RepairMapper:
    """Map detected bottlenecks to prioritised repair strategies.

    Parameters
    ----------
    max_repairs_per_bottleneck : int
        Maximum number of repair suggestions per bottleneck.
    contextualise : bool
        If True, tailor repair text using evidence values.
    """

    max_repairs_per_bottleneck: int = 5
    contextualise: bool = True

    # ── Public API --------------------------------------------------------

    def map_to_repairs(self, bottleneck: BottleneckResult) -> list[str]:
        """Return prioritised repair strategies for *bottleneck*.

        Parameters
        ----------
        bottleneck : BottleneckResult

        Returns
        -------
        list[str]
            Ordered list of repair strategy descriptions.
        """
        btype = bottleneck.bottleneck_type

        # Get base strategies
        strategies = self._get_strategies(btype)

        # Prioritise based on severity
        strategies = self._prioritise(strategies, bottleneck.severity)

        # Contextualise with evidence
        if self.contextualise:
            strategies = self._contextualise(strategies, bottleneck)

        return strategies[: self.max_repairs_per_bottleneck]

    def map_all(self, bottlenecks: list[BottleneckResult]) -> dict[str, list[str]]:
        """Map all bottlenecks to repairs, keyed by bottleneck description.

        Parameters
        ----------
        bottlenecks : list[BottleneckResult]

        Returns
        -------
        dict[str, list[str]]
        """
        result: dict[str, list[str]] = {}
        for b in bottlenecks:
            key = f"{b.bottleneck_type.value}@{','.join(b.affected_states[:3])}"
            result[key] = self.map_to_repairs(b)
        return result

    # ── Strategy retrieval ------------------------------------------------

    def _get_strategies(self, btype: BottleneckType) -> list[str]:
        """Return the full list of strategies for *btype*."""
        return list(REPAIR_STRATEGIES.get(btype, []))

    def _perceptual_repairs(self) -> list[str]:
        """Return perceptual overload repair strategies."""
        return list(_PERCEPTUAL_REPAIRS)

    def _choice_repairs(self) -> list[str]:
        """Return choice paralysis repair strategies."""
        return list(_CHOICE_REPAIRS)

    def _motor_repairs(self) -> list[str]:
        """Return motor difficulty repair strategies."""
        return list(_MOTOR_REPAIRS)

    def _memory_repairs(self) -> list[str]:
        """Return memory decay repair strategies."""
        return list(_MEMORY_REPAIRS)

    def _interference_repairs(self) -> list[str]:
        """Return cross-channel interference repair strategies."""
        return list(_INTERFERENCE_REPAIRS)

    # ── Prioritisation ----------------------------------------------------

    def _prioritise(self, strategies: list[str], severity: Severity) -> list[str]:
        """Re-order strategies based on severity.

        For CRITICAL/HIGH severity, quick-win strategies are promoted.
        For LOW/INFO severity, longer-term strategies are acceptable.

        Parameters
        ----------
        strategies : list[str]
        severity : Severity

        Returns
        -------
        list[str]
        """
        if severity in (Severity.CRITICAL, Severity.HIGH):
            # Promote strategies that mention 'increase', 'reduce', 'add'
            # (quick fixes) over those that mention 'redesign', 'replace'
            quick_keywords = {"increase", "reduce", "add", "remove", "provide"}
            slow_keywords = {"redesign", "replace", "implement", "wizard"}

            def priority_score(s: str) -> int:
                lower = s.lower()
                score = 0
                for kw in quick_keywords:
                    if kw in lower:
                        score -= 1
                for kw in slow_keywords:
                    if kw in lower:
                        score += 1
                return score

            strategies = sorted(strategies, key=priority_score)

        return strategies

    # ── Contextualisation -------------------------------------------------

    def _contextualise(
        self,
        strategies: list[str],
        bottleneck: BottleneckResult,
    ) -> list[str]:
        """Tailor strategy text using bottleneck evidence.

        For example, if the target width is known, replace the generic
        "increase target size" with "increase target from 18px to 44px".

        Parameters
        ----------
        strategies : list[str]
        bottleneck : BottleneckResult

        Returns
        -------
        list[str]
        """
        evidence = bottleneck.evidence
        result: list[str] = []

        for s in strategies:
            # Motor: contextualise target size
            if "target size" in s.lower() and "target_width" in evidence:
                width = evidence["target_width"]
                if width < 44.0:
                    s = f"{s} (current size: {width:.0f}px)"

            # Motor: contextualise distance
            if "distance" in s.lower() and "target_distance" in evidence:
                dist = evidence["target_distance"]
                s = f"{s} (current distance: {dist:.0f}px)"

            # Choice: contextualise number of options
            if "options" in s.lower() and "effective_choices" in evidence:
                n_eff = evidence["effective_choices"]
                s = f"{s} (current: ~{n_eff:.0f} effective choices)"

            # Perceptual: contextualise element count
            if "elements" in s.lower() and "n_elements" in evidence:
                n_elem = evidence["n_elements"]
                s = f"{s} (current: {n_elem:.0f} elements)"

            # Memory: contextualise WM load
            if "chunk" in s.lower() and "working_memory_load" in evidence:
                load = evidence["working_memory_load"]
                s = f"{s} (current load: {load:.0f} items)"

            result.append(s)

        return result
