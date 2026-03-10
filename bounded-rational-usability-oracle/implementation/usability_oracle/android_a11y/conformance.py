"""
usability_oracle.android_a11y.conformance — Android accessibility conformance.

Checks Android view hierarchies for common accessibility issues
per Google's Android Accessibility Guidelines.

Reference: https://developer.android.com/guide/topics/ui/accessibility
           https://support.google.com/accessibility/android/answer/6376559
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set

from usability_oracle.android_a11y.types import (
    AndroidClassName,
    AndroidNode,
    ViewHierarchy,
)
from usability_oracle.aria.types import ConformanceLevel, ConformanceResult


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

# Minimum touch target size in dp (Material Design guidelines)
_MIN_TOUCH_TARGET_DP = 48

# Default screen density (mdpi baseline: 1dp = 1px at 160dpi)
_DEFAULT_DENSITY = 2.75  # ~xxhdpi (440dpi)

# Classes that should have content descriptions
_IMAGE_CLASSES: FrozenSet[str] = frozenset({
    AndroidClassName.IMAGE_VIEW.value,
    AndroidClassName.IMAGE_BUTTON.value,
})

# Interactive classes that must be focusable
_INTERACTIVE_CLASSES: FrozenSet[str] = frozenset({
    AndroidClassName.BUTTON.value,
    AndroidClassName.IMAGE_BUTTON.value,
    AndroidClassName.CHECK_BOX.value,
    AndroidClassName.RADIO_BUTTON.value,
    AndroidClassName.TOGGLE_BUTTON.value,
    AndroidClassName.SWITCH.value,
    AndroidClassName.SEEK_BAR.value,
    AndroidClassName.SPINNER.value,
    AndroidClassName.EDIT_TEXT.value,
})


# ═══════════════════════════════════════════════════════════════════════════
# Check functions
# ═══════════════════════════════════════════════════════════════════════════

def check_content_descriptions(tree: ViewHierarchy) -> List[ConformanceResult]:
    """Check that ImageViews have content descriptions.

    Android Accessibility Guidelines: All meaningful images must
    have ``contentDescription`` set, or be marked as not important
    for accessibility.

    Parameters:
        tree: Parsed :class:`ViewHierarchy`.

    Returns:
        List of :class:`ConformanceResult` for image nodes.
    """
    results: list[ConformanceResult] = []

    for node in tree.nodes.values():
        if node.class_name not in _IMAGE_CLASSES:
            continue
        if not node.is_important_for_accessibility:
            continue
        if not node.is_visible_to_user:
            continue

        if not node.description.accessible_name:
            results.append(ConformanceResult(
                node_id=node.node_id,
                role=node.class_name,
                level=ConformanceLevel.VIOLATION,
                violations=(
                    f"ImageView '{node.node_id}' has no content description. "
                    f"Set android:contentDescription or mark as "
                    f"android:importantForAccessibility='no'",
                ),
            ))
        else:
            results.append(ConformanceResult(
                node_id=node.node_id,
                role=node.class_name,
                level=ConformanceLevel.CONFORMING,
            ))

    return results


def check_touch_target_size(
    tree: ViewHierarchy,
    density: float = _DEFAULT_DENSITY,
) -> List[ConformanceResult]:
    """Check that interactive elements meet minimum 48dp touch target size.

    Material Design Accessibility Guidelines: Touch targets should be
    at least 48×48dp for comfortable interaction.

    Parameters:
        tree: Parsed :class:`ViewHierarchy`.
        density: Screen density multiplier (px per dp).

    Returns:
        List of :class:`ConformanceResult` for undersized targets.
    """
    results: list[ConformanceResult] = []
    min_px = _MIN_TOUCH_TARGET_DP * density

    for node in tree.nodes.values():
        if not node.is_clickable and not node.is_checkable:
            continue
        if not node.is_visible_to_user:
            continue

        width_px = node.bounds.screen_width
        height_px = node.bounds.screen_height

        width_dp = width_px / density
        height_dp = height_px / density

        if width_dp < _MIN_TOUCH_TARGET_DP or height_dp < _MIN_TOUCH_TARGET_DP:
            results.append(ConformanceResult(
                node_id=node.node_id,
                role=node.class_name,
                level=ConformanceLevel.WARNING,
                warnings=(
                    f"Touch target size {width_dp:.0f}×{height_dp:.0f}dp "
                    f"is below the recommended {_MIN_TOUCH_TARGET_DP}×"
                    f"{_MIN_TOUCH_TARGET_DP}dp minimum",
                ),
            ))
        else:
            results.append(ConformanceResult(
                node_id=node.node_id,
                role=node.class_name,
                level=ConformanceLevel.CONFORMING,
            ))

    return results


def check_text_contrast(tree: ViewHierarchy) -> List[ConformanceResult]:
    """Perform structural text contrast checks.

    This is a best-effort structural check since we lack pixel data.
    Flags text elements that have no visible text or that appear to
    lack contrast information.

    Parameters:
        tree: Parsed :class:`ViewHierarchy`.

    Returns:
        List of :class:`ConformanceResult` for contrast concerns.
    """
    results: list[ConformanceResult] = []

    for node in tree.nodes.values():
        if node.class_name != AndroidClassName.TEXT_VIEW.value:
            continue
        if not node.is_visible_to_user:
            continue
        if not node.description.text:
            continue

        # Structural check: very small text in very small containers
        # may indicate contrast issues
        if node.bounds.screen_height < 10 and node.bounds.screen_width < 10:
            results.append(ConformanceResult(
                node_id=node.node_id,
                role=node.class_name,
                level=ConformanceLevel.WARNING,
                warnings=(
                    "Text element has very small bounds "
                    f"({node.bounds.screen_width}×{node.bounds.screen_height}px), "
                    "may be invisible or have contrast issues",
                ),
            ))

    return results


def check_focusable_elements(tree: ViewHierarchy) -> List[ConformanceResult]:
    """Check that interactive elements are focusable.

    Android Accessibility: All interactive elements must be
    reachable via accessibility focus (TalkBack/Switch Access).

    Parameters:
        tree: Parsed :class:`ViewHierarchy`.

    Returns:
        List of :class:`ConformanceResult` for non-focusable interactive elements.
    """
    results: list[ConformanceResult] = []

    for node in tree.nodes.values():
        if node.class_name not in _INTERACTIVE_CLASSES:
            continue
        if not node.is_enabled:
            continue
        if not node.is_visible_to_user:
            continue

        if not node.is_focusable and not node.is_clickable:
            results.append(ConformanceResult(
                node_id=node.node_id,
                role=node.class_name,
                level=ConformanceLevel.VIOLATION,
                violations=(
                    f"Interactive element '{node.class_name}' is not "
                    f"focusable. Set android:focusable='true'",
                ),
            ))
        else:
            results.append(ConformanceResult(
                node_id=node.node_id,
                role=node.class_name,
                level=ConformanceLevel.CONFORMING,
            ))

    return results


def check_traversal_order(tree: ViewHierarchy) -> List[ConformanceResult]:
    """Check for logical focus traversal order.

    Verifies that the visual order (top-to-bottom, left-to-right)
    roughly matches the logical focus traversal order. Significant
    mismatches may confuse screen reader users.

    Parameters:
        tree: Parsed :class:`ViewHierarchy`.

    Returns:
        List of :class:`ConformanceResult` for traversal order issues.
    """
    results: list[ConformanceResult] = []

    # Collect focusable nodes in document order (by depth-first child_ids)
    doc_order: list[AndroidNode] = []
    _collect_focusable_dfs(tree.root_id, tree.nodes, doc_order)

    if len(doc_order) < 2:
        return results

    # Check that visual position doesn't go backwards significantly
    prev_top = -1
    prev_left = -1
    for node in doc_order:
        top = node.bounds.screen_top
        left = node.bounds.screen_left

        # Allow same row (within 20px tolerance)
        if top < prev_top - 20:
            results.append(ConformanceResult(
                node_id=node.node_id,
                role=node.class_name,
                level=ConformanceLevel.WARNING,
                warnings=(
                    f"Focus traversal may be out of visual order: "
                    f"element at y={top} comes after element at y={prev_top}",
                ),
            ))

        prev_top = top
        prev_left = left

    return results


def check_redundant_descriptions(tree: ViewHierarchy) -> List[ConformanceResult]:
    """Check for redundant accessibility descriptions.

    Flags cases where the content description duplicates the role
    or class information (e.g. "button button", "image image"),
    which causes screen readers to announce redundant information.

    Parameters:
        tree: Parsed :class:`ViewHierarchy`.

    Returns:
        List of :class:`ConformanceResult` for redundant descriptions.
    """
    results: list[ConformanceResult] = []

    # Role words that should not appear in content descriptions
    role_words = {
        AndroidClassName.BUTTON.value: {"button"},
        AndroidClassName.IMAGE_BUTTON.value: {"button", "image button"},
        AndroidClassName.IMAGE_VIEW.value: {"image"},
        AndroidClassName.CHECK_BOX.value: {"checkbox", "check box"},
        AndroidClassName.RADIO_BUTTON.value: {"radio button", "radio"},
        AndroidClassName.EDIT_TEXT.value: {"edit text", "text field", "text box"},
        AndroidClassName.TOGGLE_BUTTON.value: {"toggle"},
        AndroidClassName.SWITCH.value: {"switch"},
    }

    for node in tree.nodes.values():
        if not node.description.content_description:
            continue

        desc_lower = node.description.content_description.lower().strip()
        redundant_words = role_words.get(node.class_name, set())

        for word in redundant_words:
            if desc_lower == word or desc_lower.startswith(word + " ") or desc_lower.endswith(" " + word):
                results.append(ConformanceResult(
                    node_id=node.node_id,
                    role=node.class_name,
                    level=ConformanceLevel.WARNING,
                    warnings=(
                        f"Content description '{node.description.content_description}' "
                        f"contains redundant role information '{word}'. "
                        f"TalkBack already announces the element type",
                    ),
                ))
                break

    return results


def run_all_checks(
    tree: ViewHierarchy,
    density: float = _DEFAULT_DENSITY,
) -> List[ConformanceResult]:
    """Run all Android accessibility conformance checks.

    Parameters:
        tree: Parsed :class:`ViewHierarchy`.
        density: Screen density for touch target size calculation.

    Returns:
        Comprehensive list of :class:`ConformanceResult`.
    """
    results: list[ConformanceResult] = []

    results.extend(check_content_descriptions(tree))
    results.extend(check_touch_target_size(tree, density=density))
    results.extend(check_text_contrast(tree))
    results.extend(check_focusable_elements(tree))
    results.extend(check_traversal_order(tree))
    results.extend(check_redundant_descriptions(tree))

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════

def _collect_focusable_dfs(
    node_id: str,
    nodes: Dict[str, AndroidNode],
    result: list[AndroidNode],
) -> None:
    """Depth-first collection of focusable nodes."""
    node = nodes.get(node_id)
    if node is None:
        return

    if node.is_focusable and node.is_visible_to_user:
        result.append(node)

    for child_id in node.child_ids:
        _collect_focusable_dfs(child_id, nodes, result)
