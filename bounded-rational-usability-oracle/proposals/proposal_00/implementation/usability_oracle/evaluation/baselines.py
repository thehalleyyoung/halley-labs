"""
usability_oracle.evaluation.baselines — Baseline comparators.

Implements simple baseline methods for regression detection so that the
oracle's performance can be compared against trivial or classical methods.
"""

from __future__ import annotations

import math
import random
from typing import Any, Optional

from usability_oracle.accessibility.models import AccessibilityNode, AccessibilityTree
from usability_oracle.core.enums import RegressionVerdict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_nodes(tree: AccessibilityTree) -> int:
    """Count total nodes in an accessibility tree."""
    count = 0

    def _walk(node: AccessibilityNode) -> None:
        nonlocal count
        count += 1
        for child in node.children:
            _walk(child)

    _walk(tree.root)
    return count


def _max_depth(tree: AccessibilityTree) -> int:
    """Compute maximum depth of an accessibility tree."""
    def _depth(node: AccessibilityNode) -> int:
        if not node.children:
            return 0
        return 1 + max(_depth(c) for c in node.children)

    return _depth(tree.root)


def _interactive_nodes(tree: AccessibilityTree) -> list[AccessibilityNode]:
    """Collect interactive nodes (buttons, links, inputs, etc.)."""
    interactive_roles = {"button", "link", "textfield", "checkbox", "radio", "menuitem", "tab"}
    result: list[AccessibilityNode] = []

    def _walk(node: AccessibilityNode) -> None:
        if node.role.lower().replace("_", "") in interactive_roles:
            result.append(node)
        for child in node.children:
            _walk(child)

    _walk(tree.root)
    return result


def _estimate_klm_time(tree: AccessibilityTree, task_steps: int = 5) -> float:
    """Very rough KLM time estimate in seconds.

    Uses simplified KLM operators:
      K (keystroke) = 0.2s, P (pointing) = 1.1s, M (mental) = 1.35s, H (homing) = 0.4s
    """
    interactive = _interactive_nodes(tree)
    n_targets = max(len(interactive), 1)
    # Assume each task step involves: M + P + K
    pointing_time = 1.1
    if interactive:
        # Crude Fitts' law: larger targets are faster
        avg_area = 0.0
        for node in interactive:
            if node.bounding_box:
                avg_area += node.bounding_box.width * node.bounding_box.height
        avg_area /= len(interactive)
        if avg_area > 0:
            # Fitts' index of difficulty approximation
            id_val = math.log2(2.0 * 200.0 / max(math.sqrt(avg_area), 1.0))
            pointing_time = 0.1 + 0.1 * max(id_val, 0)

    mental_time = 1.35
    keystroke_time = 0.2
    homing_time = 0.4

    total = task_steps * (mental_time + pointing_time + keystroke_time) + homing_time
    # Penalty for Hick-Hyman: log2(n_targets + 1) extra mental time
    hick_penalty = mental_time * math.log2(n_targets + 1) * 0.3
    total += hick_penalty
    return total


# ---------------------------------------------------------------------------
# BaselineComparator
# ---------------------------------------------------------------------------

class BaselineComparator:
    """Collection of baseline methods for usability regression detection.

    Each method accepts two accessibility trees (before/after) and returns
    a :class:`RegressionVerdict`.  These baselines serve as lower bounds
    for oracle performance.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Element count baseline
    # ------------------------------------------------------------------

    def element_count_baseline(
        self,
        tree_a: AccessibilityTree,
        tree_b: AccessibilityTree,
        threshold: float = 0.2,
    ) -> RegressionVerdict:
        """Compare total element counts.

        A significant increase in element count is treated as a regression
        (more clutter), a decrease as an improvement (simplified UI).
        """
        count_a = _count_nodes(tree_a)
        count_b = _count_nodes(tree_b)
        if count_a == 0:
            return RegressionVerdict.INCONCLUSIVE
        ratio = (count_b - count_a) / count_a
        if ratio > threshold:
            return RegressionVerdict.REGRESSION
        if ratio < -threshold:
            return RegressionVerdict.IMPROVEMENT
        return RegressionVerdict.NEUTRAL

    # ------------------------------------------------------------------
    # Depth baseline
    # ------------------------------------------------------------------

    def depth_baseline(
        self,
        tree_a: AccessibilityTree,
        tree_b: AccessibilityTree,
        threshold: int = 2,
    ) -> RegressionVerdict:
        """Compare maximum tree depth.

        A significantly deeper tree implies more navigation complexity.
        """
        depth_a = _max_depth(tree_a)
        depth_b = _max_depth(tree_b)
        diff = depth_b - depth_a
        if diff > threshold:
            return RegressionVerdict.REGRESSION
        if diff < -threshold:
            return RegressionVerdict.IMPROVEMENT
        return RegressionVerdict.NEUTRAL

    # ------------------------------------------------------------------
    # KLM baseline
    # ------------------------------------------------------------------

    def klm_baseline(
        self,
        tree_a: AccessibilityTree,
        tree_b: AccessibilityTree,
        task: Any = None,
        threshold: float = 0.15,
    ) -> RegressionVerdict:
        """Basic Keystroke-Level Model comparison.

        Estimates task-completion time for both versions and compares.
        """
        steps = 5
        if task and hasattr(task, "n_steps"):
            steps = task.n_steps

        time_a = _estimate_klm_time(tree_a, task_steps=steps)
        time_b = _estimate_klm_time(tree_b, task_steps=steps)

        if time_a == 0:
            return RegressionVerdict.INCONCLUSIVE
        ratio = (time_b - time_a) / time_a
        if ratio > threshold:
            return RegressionVerdict.REGRESSION
        if ratio < -threshold:
            return RegressionVerdict.IMPROVEMENT
        return RegressionVerdict.NEUTRAL

    # ------------------------------------------------------------------
    # Random baseline
    # ------------------------------------------------------------------

    def random_baseline(self) -> RegressionVerdict:
        """Random coin flip.  Expected accuracy ≈ 1/|verdicts|."""
        return self._rng.choice([
            RegressionVerdict.REGRESSION,
            RegressionVerdict.IMPROVEMENT,
            RegressionVerdict.NEUTRAL,
        ])

    # ------------------------------------------------------------------
    # Always-neutral baseline
    # ------------------------------------------------------------------

    @staticmethod
    def always_neutral() -> RegressionVerdict:
        """Always predicts NEUTRAL.  Good accuracy when regressions are rare."""
        return RegressionVerdict.NEUTRAL

    # ------------------------------------------------------------------
    # Run all baselines
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Interactive count baseline
    # ------------------------------------------------------------------

    def interactive_count_baseline(
        self,
        tree_a: AccessibilityTree,
        tree_b: AccessibilityTree,
        threshold: float = 0.25,
    ) -> RegressionVerdict:
        """Compare counts of interactive elements.

        More interactive elements may indicate choice paralysis;
        fewer may indicate lost functionality.
        """
        count_a = len(_interactive_nodes(tree_a))
        count_b = len(_interactive_nodes(tree_b))
        if count_a == 0:
            return RegressionVerdict.INCONCLUSIVE
        ratio = (count_b - count_a) / count_a
        if ratio > threshold:
            return RegressionVerdict.REGRESSION
        if ratio < -threshold:
            return RegressionVerdict.IMPROVEMENT
        return RegressionVerdict.NEUTRAL

    # ------------------------------------------------------------------
    # Target size baseline (Fitts' law)
    # ------------------------------------------------------------------

    def target_size_baseline(
        self,
        tree_a: AccessibilityTree,
        tree_b: AccessibilityTree,
        min_target_area: float = 44 * 44,
    ) -> RegressionVerdict:
        """Compare minimum target sizes (WCAG 2.5.5).

        If targets shrink below the minimum, that's a regression.
        """
        def _min_target_area(tree: AccessibilityTree) -> float:
            areas = []
            for node in _interactive_nodes(tree):
                if node.bounding_box:
                    area = node.bounding_box.width * node.bounding_box.height
                    areas.append(area)
            return min(areas) if areas else 0.0

        min_a = _min_target_area(tree_a)
        min_b = _min_target_area(tree_b)

        if min_a >= min_target_area and min_b < min_target_area:
            return RegressionVerdict.REGRESSION
        if min_a < min_target_area and min_b >= min_target_area:
            return RegressionVerdict.IMPROVEMENT
        if min_b < min_a * 0.7:
            return RegressionVerdict.REGRESSION
        if min_b > min_a * 1.3:
            return RegressionVerdict.IMPROVEMENT
        return RegressionVerdict.NEUTRAL

    # ------------------------------------------------------------------
    # Heading structure baseline
    # ------------------------------------------------------------------

    def heading_structure_baseline(
        self,
        tree_a: AccessibilityTree,
        tree_b: AccessibilityTree,
    ) -> RegressionVerdict:
        """Compare heading structure for navigation quality."""
        def _heading_count(tree: AccessibilityTree) -> int:
            count = 0
            def _walk(node: AccessibilityNode) -> None:
                nonlocal count
                if node.role.lower() in ("heading",):
                    count += 1
                for child in node.children:
                    _walk(child)
            _walk(tree.root)
            return count

        h_a = _heading_count(tree_a)
        h_b = _heading_count(tree_b)

        # Lost headings = worse navigation
        if h_a > 0 and h_b == 0:
            return RegressionVerdict.REGRESSION
        if h_b > h_a + 2:
            return RegressionVerdict.IMPROVEMENT
        if h_b < h_a - 2:
            return RegressionVerdict.REGRESSION
        return RegressionVerdict.NEUTRAL

    # ------------------------------------------------------------------
    # Label coverage baseline
    # ------------------------------------------------------------------

    def label_coverage_baseline(
        self,
        tree_a: AccessibilityTree,
        tree_b: AccessibilityTree,
        threshold: float = 0.1,
    ) -> RegressionVerdict:
        """Compare fraction of interactive elements that have labels."""
        def _label_coverage(tree: AccessibilityTree) -> float:
            interactive = _interactive_nodes(tree)
            if not interactive:
                return 1.0
            labeled = sum(1 for n in interactive if n.name and n.name.strip())
            return labeled / len(interactive)

        cov_a = _label_coverage(tree_a)
        cov_b = _label_coverage(tree_b)
        diff = cov_b - cov_a

        if diff < -threshold:
            return RegressionVerdict.REGRESSION
        if diff > threshold:
            return RegressionVerdict.IMPROVEMENT
        return RegressionVerdict.NEUTRAL

    # ------------------------------------------------------------------
    # Run all baselines
    # ------------------------------------------------------------------

    def run_all(
        self,
        tree_a: AccessibilityTree,
        tree_b: AccessibilityTree,
        task: Any = None,
    ) -> dict[str, RegressionVerdict]:
        """Run every baseline and return a dict of results."""
        return {
            "element_count": self.element_count_baseline(tree_a, tree_b),
            "depth": self.depth_baseline(tree_a, tree_b),
            "klm": self.klm_baseline(tree_a, tree_b, task),
            "interactive_count": self.interactive_count_baseline(tree_a, tree_b),
            "target_size": self.target_size_baseline(tree_a, tree_b),
            "heading_structure": self.heading_structure_baseline(tree_a, tree_b),
            "label_coverage": self.label_coverage_baseline(tree_a, tree_b),
            "random": self.random_baseline(),
            "always_neutral": self.always_neutral(),
        }

    # ------------------------------------------------------------------
    # Majority vote ensemble
    # ------------------------------------------------------------------

    def majority_vote(
        self,
        tree_a: AccessibilityTree,
        tree_b: AccessibilityTree,
        task: Any = None,
    ) -> RegressionVerdict:
        """Ensemble baseline using majority vote across all baselines."""
        results = self.run_all(tree_a, tree_b, task)
        votes: dict[str, int] = {}
        for verdict in results.values():
            key = verdict.value
            votes[key] = votes.get(key, 0) + 1

        if not votes:
            return RegressionVerdict.NEUTRAL
        winner = max(votes, key=votes.get)  # type: ignore[arg-type]
        return RegressionVerdict(winner)
