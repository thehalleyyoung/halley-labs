"""
usability_oracle.benchmarks.mutations — Known-bottleneck mutations.

Applies controlled mutations to accessibility trees to inject specific
types of usability bottlenecks at configurable severity levels.  This
enables ground-truth benchmark generation where the expected bottleneck
type and severity are known *a priori*.
"""

from __future__ import annotations

import copy
import random
import uuid
from typing import Any

from usability_oracle.accessibility.models import (
    AccessibilityNode,
    AccessibilityState,
    AccessibilityTree,
    BoundingBox,
)
from usability_oracle.core.enums import AccessibilityRole, BottleneckType


def _uid() -> str:
    return uuid.uuid4().hex[:8]


def _deep_copy_tree(tree: AccessibilityTree) -> AccessibilityTree:
    """Create a deep copy of an accessibility tree."""
    new_root = copy.deepcopy(tree.root)
    idx: dict[str, AccessibilityNode] = {}

    def _reindex(node: AccessibilityNode) -> None:
        idx[node.id] = node
        for child in node.children:
            _reindex(child)

    _reindex(new_root)
    return AccessibilityTree(root=new_root, node_index=idx, metadata=dict(tree.metadata))


def _collect_nodes(tree: AccessibilityTree) -> list[AccessibilityNode]:
    """Flat list of all nodes in DFS order."""
    result: list[AccessibilityNode] = []

    def _walk(node: AccessibilityNode) -> None:
        result.append(node)
        for child in node.children:
            _walk(child)

    _walk(tree.root)
    return result


def _make_filler_node(
    role: str,
    name: str,
    bbox: BoundingBox,
    depth: int = 0,
) -> AccessibilityNode:
    return AccessibilityNode(
        id=_uid(),
        role=role,
        name=name,
        bounding_box=bbox,
        properties={},
        state=AccessibilityState(),
        children=[],
        depth=depth,
    )


# ---------------------------------------------------------------------------
# MutationGenerator
# ---------------------------------------------------------------------------

class MutationGenerator:
    """Apply controlled mutations that inject known usability bottlenecks.

    Each method returns a *new* accessibility tree (the original is not
    modified).  The *severity* parameter (0.0–1.0) controls how extreme
    the mutation is.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Perceptual overload
    # ------------------------------------------------------------------

    def apply_perceptual_overload(
        self,
        tree: AccessibilityTree,
        severity: float = 0.5,
    ) -> AccessibilityTree:
        """Add excessive elements and reduce grouping.

        * Injects extra generic/image nodes proportional to severity.
        * Flattens some region groups into the parent.
        """
        new_tree = _deep_copy_tree(tree)
        nodes = _collect_nodes(new_tree)
        n_add = max(1, int(len(nodes) * severity * 2))

        # Add clutter nodes to random parents
        parents = [n for n in nodes if n.children or n.role in (AccessibilityRole.REGION.value, AccessibilityRole.FORM.value)]
        if not parents:
            parents = [new_tree.root]

        for i in range(n_add):
            parent = self._rng.choice(parents)
            pbox = parent.bounding_box or BoundingBox(x=0, y=0, width=200, height=200)
            x = pbox.x + self._rng.uniform(0, max(pbox.width - 30, 0))
            y = pbox.y + self._rng.uniform(0, max(pbox.height - 20, 0))
            bbox = BoundingBox(x=round(x, 1), y=round(y, 1),
                               width=round(self._rng.uniform(20, 80), 1),
                               height=round(self._rng.uniform(14, 40), 1))
            role = self._rng.choice([AccessibilityRole.GENERIC.value, AccessibilityRole.IMAGE.value])
            filler = _make_filler_node(role, f"clutter-{i}", bbox, parent.depth + 1)
            parent.children.append(filler)
            new_tree.node_index[filler.id] = filler

        # Flatten some groups if severity is high
        if severity > 0.4:
            n_flatten = max(1, int(len(nodes) * (severity - 0.4)))
            regions = [n for n in _collect_nodes(new_tree) if n.role == AccessibilityRole.REGION.value and n.children]
            self._rng.shuffle(regions)
            for region in regions[:n_flatten]:
                if region.parent_id and region.parent_id in new_tree.node_index:
                    parent = new_tree.node_index[region.parent_id]
                    idx = next((i for i, c in enumerate(parent.children) if c.id == region.id), None)
                    if idx is not None:
                        promoted = region.children[:]
                        for child in promoted:
                            child.parent_id = parent.id
                            child.depth = region.depth
                        parent.children[idx:idx + 1] = promoted

        new_tree.metadata["mutation"] = BottleneckType.PERCEPTUAL_OVERLOAD.value
        new_tree.metadata["mutation_severity"] = severity
        return new_tree

    # ------------------------------------------------------------------
    # Choice paralysis
    # ------------------------------------------------------------------

    def apply_choice_paralysis(
        self,
        tree: AccessibilityTree,
        severity: float = 0.5,
    ) -> AccessibilityTree:
        """Add many similar options and flatten menus.

        * Duplicates button/link/menu-item nodes.
        * Converts hierarchical menus into flat lists.
        """
        new_tree = _deep_copy_tree(tree)
        nodes = _collect_nodes(new_tree)
        actionable_roles = {
            AccessibilityRole.BUTTON.value,
            AccessibilityRole.LINK.value,
            AccessibilityRole.MENUITEM.value,
        }
        actionable = [n for n in nodes if n.role in actionable_roles]
        n_dup = max(1, int(len(actionable) * severity * 3))

        for i in range(n_dup):
            if not actionable:
                break
            source = self._rng.choice(actionable)
            parent_id = source.parent_id or new_tree.root.id
            parent = new_tree.node_index.get(parent_id, new_tree.root)
            bbox = source.bounding_box or BoundingBox(x=0, y=0, width=80, height=30)
            offset_y = bbox.height * (i + 1)
            new_bbox = BoundingBox(x=bbox.x, y=bbox.y + offset_y, width=bbox.width, height=bbox.height)
            dup = _make_filler_node(source.role, f"{source.name} ({i + 2})", new_bbox, source.depth)
            dup.parent_id = parent.id
            parent.children.append(dup)
            new_tree.node_index[dup.id] = dup

        # Flatten menus
        if severity > 0.3:
            menus = [n for n in _collect_nodes(new_tree) if n.role == AccessibilityRole.MENU.value and n.children]
            for menu in menus:
                if menu.parent_id and menu.parent_id in new_tree.node_index:
                    parent = new_tree.node_index[menu.parent_id]
                    idx = next((i for i, c in enumerate(parent.children) if c.id == menu.id), None)
                    if idx is not None:
                        flat_items = []
                        self._flatten_children(menu, flat_items)
                        for item in flat_items:
                            item.parent_id = parent.id
                            item.depth = menu.depth
                        parent.children[idx:idx + 1] = flat_items

        new_tree.metadata["mutation"] = BottleneckType.CHOICE_PARALYSIS.value
        new_tree.metadata["mutation_severity"] = severity
        return new_tree

    def _flatten_children(self, node: AccessibilityNode, out: list[AccessibilityNode]) -> None:
        for child in node.children:
            out.append(child)
            if child.children:
                self._flatten_children(child, out)
                child.children = []

    # ------------------------------------------------------------------
    # Motor difficulty
    # ------------------------------------------------------------------

    def apply_motor_difficulty(
        self,
        tree: AccessibilityTree,
        severity: float = 0.5,
    ) -> AccessibilityTree:
        """Shrink targets and increase distances between interactive elements."""
        new_tree = _deep_copy_tree(tree)
        interactive_roles = {
            AccessibilityRole.BUTTON.value,
            AccessibilityRole.LINK.value,
            AccessibilityRole.TEXTFIELD.value,
            AccessibilityRole.CHECKBOX.value,
            AccessibilityRole.RADIO.value,
            AccessibilityRole.MENUITEM.value,
        }
        nodes = _collect_nodes(new_tree)
        interactive = [n for n in nodes if n.role in interactive_roles and n.bounding_box]

        shrink_factor = max(0.1, 1.0 - severity * 0.85)
        spread_factor = 1.0 + severity * 4.0

        for node in interactive:
            bbox = node.bounding_box
            assert bbox is not None
            # Shrink target size
            new_w = max(4, bbox.width * shrink_factor)
            new_h = max(4, bbox.height * shrink_factor)
            # Move away from centre to increase distances
            cx, cy = bbox.x + bbox.width / 2, bbox.y + bbox.height / 2
            new_x = cx * spread_factor - new_w / 2
            new_y = cy * spread_factor - new_h / 2
            node.bounding_box = BoundingBox(
                x=round(max(0, new_x), 1),
                y=round(max(0, new_y), 1),
                width=round(new_w, 1),
                height=round(new_h, 1),
            )

        new_tree.metadata["mutation"] = BottleneckType.MOTOR_DIFFICULTY.value
        new_tree.metadata["mutation_severity"] = severity
        return new_tree

    # ------------------------------------------------------------------
    # Memory decay
    # ------------------------------------------------------------------

    def apply_memory_decay(
        self,
        tree: AccessibilityTree,
        severity: float = 0.5,
    ) -> AccessibilityTree:
        """Add steps and remove state indicators.

        * Injects intermediate confirmation/loading steps.
        * Removes status labels and progress indicators.
        """
        new_tree = _deep_copy_tree(tree)
        nodes = _collect_nodes(new_tree)
        n_steps = max(1, int(severity * 6))

        # Add intermediate steps
        for i in range(n_steps):
            parent = self._rng.choice(nodes) if nodes else new_tree.root
            pbox = parent.bounding_box or BoundingBox(x=0, y=0, width=400, height=300)
            step_bbox = BoundingBox(x=pbox.x, y=pbox.y + pbox.height + 10, width=pbox.width, height=50)
            step = _make_filler_node(
                AccessibilityRole.DIALOG.value,
                f"Confirmation Step {i + 1}",
                step_bbox,
                parent.depth + 1,
            )
            confirm_btn = _make_filler_node(
                AccessibilityRole.BUTTON.value,
                "Confirm",
                BoundingBox(x=step_bbox.x + 10, y=step_bbox.y + 10, width=80, height=30),
                parent.depth + 2,
            )
            step.children.append(confirm_btn)
            step.parent_id = parent.id
            parent.children.append(step)
            new_tree.node_index[step.id] = step
            new_tree.node_index[confirm_btn.id] = confirm_btn

        # Remove state indicators (hide checked/selected states)
        if severity > 0.3:
            for node in _collect_nodes(new_tree):
                if node.state and (node.state.checked is not None or node.state.selected):
                    node.state = AccessibilityState(
                        checked=None,
                        selected=False,
                        focused=node.state.focused,
                        expanded=node.state.expanded,
                        disabled=node.state.disabled,
                        hidden=node.state.hidden,
                    )
                # Remove progress/status in properties
                node.properties.pop("aria-valuenow", None)
                node.properties.pop("aria-valuetext", None)

        new_tree.metadata["mutation"] = BottleneckType.MEMORY_DECAY.value
        new_tree.metadata["mutation_severity"] = severity
        return new_tree

    # ------------------------------------------------------------------
    # Cross-channel interference
    # ------------------------------------------------------------------

    def apply_interference(
        self,
        tree: AccessibilityTree,
        severity: float = 0.5,
    ) -> AccessibilityTree:
        """Add competing channels and overlapping modalities.

        * Adds overlapping elements in the same spatial region.
        * Injects decorative images and auto-play annotations.
        """
        new_tree = _deep_copy_tree(tree)
        nodes = _collect_nodes(new_tree)
        n_distractors = max(1, int(len(nodes) * severity * 1.5))

        for i in range(n_distractors):
            target = self._rng.choice(nodes) if nodes else new_tree.root
            tbox = target.bounding_box or BoundingBox(x=0, y=0, width=200, height=100)
            overlap_bbox = BoundingBox(
                x=tbox.x + self._rng.uniform(-20, 20),
                y=tbox.y + self._rng.uniform(-20, 20),
                width=tbox.width * self._rng.uniform(0.5, 1.5),
                height=tbox.height * self._rng.uniform(0.5, 1.5),
            )
            distractor_type = self._rng.choice([
                AccessibilityRole.IMAGE.value,
                AccessibilityRole.GENERIC.value,
            ])
            distractor = _make_filler_node(
                distractor_type,
                f"distractor-{i}",
                overlap_bbox,
                target.depth,
            )
            distractor.properties["aria-hidden"] = "false"
            distractor.properties["decorative"] = "true"

            parent_id = target.parent_id or new_tree.root.id
            parent = new_tree.node_index.get(parent_id, new_tree.root)
            distractor.parent_id = parent.id
            parent.children.append(distractor)
            new_tree.node_index[distractor.id] = distractor

        new_tree.metadata["mutation"] = BottleneckType.CROSS_CHANNEL_INTERFERENCE.value
        new_tree.metadata["mutation_severity"] = severity
        return new_tree

    # ------------------------------------------------------------------
    # Random mutation
    # ------------------------------------------------------------------

    def apply_random_mutation(
        self,
        tree: AccessibilityTree,
        seed: int | None = None,
    ) -> tuple[AccessibilityTree, str]:
        """Apply a random mutation and return ``(mutated_tree, mutation_name)``."""
        rng = random.Random(seed) if seed is not None else self._rng
        mutations = [
            (self.apply_perceptual_overload, BottleneckType.PERCEPTUAL_OVERLOAD),
            (self.apply_choice_paralysis, BottleneckType.CHOICE_PARALYSIS),
            (self.apply_motor_difficulty, BottleneckType.MOTOR_DIFFICULTY),
            (self.apply_memory_decay, BottleneckType.MEMORY_DECAY),
            (self.apply_interference, BottleneckType.CROSS_CHANNEL_INTERFERENCE),
        ]
        fn, bt = rng.choice(mutations)
        severity = rng.uniform(0.2, 0.9)
        mutated = fn(tree, severity)
        return mutated, bt.value

    # ------------------------------------------------------------------
    # Label removal mutation
    # ------------------------------------------------------------------

    def apply_label_removal(
        self,
        tree: AccessibilityTree,
        fraction: float = 0.5,
    ) -> AccessibilityTree:
        """Remove accessible names from a fraction of interactive elements.

        This simulates a common accessibility regression where labels
        are lost (e.g. after a CSS refactor or icon replacement).
        """
        new_tree = self._deep_copy_tree(tree)
        interactive = [
            n for n in new_tree.node_index.values()
            if n.role.lower() in ("button", "textbox", "link", "checkbox", "radio",
                                    "combobox", "slider", "menuitem")
        ]
        if not interactive:
            return new_tree

        n_remove = max(1, int(len(interactive) * fraction))
        targets = self._rng.sample(interactive, min(n_remove, len(interactive)))
        for node in targets:
            node.name = ""
            node.properties.pop("aria-label", None)
            node.properties.pop("aria-labelledby", None)

        new_tree.metadata["mutation"] = "label_removal"
        new_tree.metadata["mutation_severity"] = fraction
        return new_tree

    # ------------------------------------------------------------------
    # Tab-order scrambling mutation
    # ------------------------------------------------------------------

    def apply_tab_order_scramble(
        self,
        tree: AccessibilityTree,
        severity: float = 0.5,
    ) -> AccessibilityTree:
        """Assign random positive tabindex values to disrupt tab order.

        In the real DOM, positive tabindex values override the default
        document order, creating confusing keyboard navigation.
        """
        new_tree = self._deep_copy_tree(tree)
        focusable = [
            n for n in new_tree.node_index.values()
            if n.role.lower() in ("button", "textbox", "link", "checkbox",
                                    "radio", "combobox", "slider", "menuitem",
                                    "tab", "treeitem")
        ]
        if not focusable:
            return new_tree

        n_scramble = max(1, int(len(focusable) * severity))
        targets = self._rng.sample(focusable, min(n_scramble, len(focusable)))
        for node in targets:
            node.properties["tabindex"] = str(self._rng.randint(1, 100))

        new_tree.metadata["mutation"] = "tab_order_scramble"
        new_tree.metadata["mutation_severity"] = severity
        return new_tree

    # ------------------------------------------------------------------
    # Focus trap mutation
    # ------------------------------------------------------------------

    def apply_focus_trap(
        self,
        tree: AccessibilityTree,
        trap_depth: int = 3,
    ) -> AccessibilityTree:
        """Create a focus trap — a subtree that captures keyboard focus.

        Inserts a group of deeply nested focusable elements that
        conceptually represent a widget with no keyboard escape.
        """
        new_tree = self._deep_copy_tree(tree)
        # Pick a random interactive node to be the trap entrance
        interactive = [
            n for n in new_tree.node_index.values()
            if n.role.lower() in ("button", "textbox", "link")
        ]
        if not interactive:
            return new_tree

        anchor = self._rng.choice(interactive)
        trap_root = AccessibilityNode(
            id="focus-trap-root",
            role="group",
            name="",
            bounding_box=anchor.bounding_box,
            properties={"aria-hidden": "false"},
            state=AccessibilityState(),
            children=[],
            depth=anchor.depth + 1,
            parent_id=anchor.id,
        )

        current = trap_root
        for d in range(trap_depth):
            child = AccessibilityNode(
                id=f"trap-level-{d}",
                role="textbox" if d == trap_depth - 1 else "group",
                name=f"Trap level {d}",
                bounding_box=anchor.bounding_box,
                properties={"tabindex": "0"},
                state=AccessibilityState(),
                children=[],
                depth=anchor.depth + 2 + d,
                parent_id=current.id,
            )
            current.children.append(child)
            new_tree.node_index[child.id] = child
            current = child

        anchor.children.append(trap_root)
        new_tree.node_index[trap_root.id] = trap_root

        new_tree.metadata["mutation"] = "focus_trap"
        new_tree.metadata["mutation_severity"] = 0.8
        return new_tree

    # ------------------------------------------------------------------
    # Contrast reduction mutation
    # ------------------------------------------------------------------

    def apply_contrast_reduction(
        self,
        tree: AccessibilityTree,
        severity: float = 0.5,
    ) -> AccessibilityTree:
        """Reduce colour contrast metadata on text elements.

        Simulates a design change that fails WCAG contrast requirements.
        """
        new_tree = self._deep_copy_tree(tree)
        text_nodes = [
            n for n in new_tree.node_index.values()
            if n.role.lower() in ("text", "heading", "label", "link", "button")
        ]
        if not text_nodes:
            return new_tree

        n_affected = max(1, int(len(text_nodes) * severity))
        targets = self._rng.sample(text_nodes, min(n_affected, len(text_nodes)))

        for node in targets:
            # Set a low contrast ratio in properties
            base_ratio = float(node.properties.get("contrast-ratio", "7.0"))
            new_ratio = max(1.0, base_ratio * (1.0 - severity * 0.8))
            node.properties["contrast-ratio"] = f"{new_ratio:.1f}"
            # Mark as failing WCAG AA if below 4.5
            if new_ratio < 4.5:
                node.properties["wcag-aa-fail"] = "true"

        new_tree.metadata["mutation"] = "contrast_reduction"
        new_tree.metadata["mutation_severity"] = severity
        return new_tree

    # ------------------------------------------------------------------
    # Missing landmark mutation
    # ------------------------------------------------------------------

    def apply_landmark_removal(
        self,
        tree: AccessibilityTree,
        fraction: float = 0.5,
    ) -> AccessibilityTree:
        """Remove landmark roles from a fraction of landmark elements.

        Simulates a redesign where semantic landmarks (navigation,
        banner, main, contentinfo) are replaced with generic divs.
        """
        new_tree = self._deep_copy_tree(tree)
        landmarks = [
            n for n in new_tree.node_index.values()
            if n.role.lower() in ("navigation", "banner", "main", "contentinfo",
                                    "complementary", "search", "region")
        ]
        if not landmarks:
            return new_tree

        n_remove = max(1, int(len(landmarks) * fraction))
        targets = self._rng.sample(landmarks, min(n_remove, len(landmarks)))
        for node in targets:
            node.properties["original_role"] = node.role
            node.role = "generic"

        new_tree.metadata["mutation"] = "landmark_removal"
        new_tree.metadata["mutation_severity"] = fraction
        return new_tree
