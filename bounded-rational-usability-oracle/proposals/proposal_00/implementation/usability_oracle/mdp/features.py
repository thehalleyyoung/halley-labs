"""
usability_oracle.mdp.features — State feature extraction from accessibility trees.

Extracts a fixed-length numeric feature vector from each MDP state by
querying the accessibility tree for structural and visual properties.
These features serve as inputs to cognitive cost models and can be used
for state aggregation (bisimulation) or policy approximation.

Feature vector (in order):
    0. visual_complexity — number of visible elements in viewport vicinity
    1. interaction_density — fraction of nearby elements that are interactive
    2. depth_in_tree — depth of the focused node
    3. n_choices — number of available actions from this state
    4. target_distance — tree distance to nearest target node
    5. working_memory_load — estimated WM items held
    6. task_progress — fraction of sub-goals completed
    7. bbox_area — normalised bounding-box area of focused node
    8. n_siblings — number of sibling nodes
    9. subtree_size — number of nodes in the focused node's subtree

References
----------
- Card, S. K., Moran, T. P. & Newell, A. (1983). *The Psychology of
  Human-Computer Interaction*. Lawrence Erlbaum.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Optional, Sequence

import numpy as np

from usability_oracle.mdp.models import State


# ---------------------------------------------------------------------------
# Feature names (fixed ordering)
# ---------------------------------------------------------------------------

_FEATURE_NAMES: list[str] = [
    "visual_complexity",
    "interaction_density",
    "depth_in_tree",
    "n_choices",
    "target_distance",
    "working_memory_load",
    "task_progress",
    "bbox_area",
    "n_siblings",
    "subtree_size",
]


# ---------------------------------------------------------------------------
# StateFeatureExtractor
# ---------------------------------------------------------------------------


class StateFeatureExtractor:
    """Extract numeric feature vectors from MDP states.

    Usage
    -----
    >>> extractor = StateFeatureExtractor(target_node_ids=["btn-submit"])
    >>> vec = extractor.extract(state, tree)
    >>> names = extractor.feature_names()
    """

    def __init__(
        self,
        target_node_ids: Optional[list[str]] = None,
        viewport_width: float = 1920.0,
        viewport_height: float = 1080.0,
    ) -> None:
        self.target_node_ids = target_node_ids or []
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self._tree_distance_cache: dict[tuple[str, str], float] = {}

    # ── Public API --------------------------------------------------------

    def extract(self, state: State, tree: Any) -> np.ndarray:
        """Extract a feature vector for *state* from *tree*.

        Parameters
        ----------
        state : State
        tree : AccessibilityTree

        Returns
        -------
        np.ndarray
            Feature vector of length ``len(feature_names())``.
        """
        node_id = state.metadata.get("node_id", state.state_id.split(":")[0])
        node = tree.get_node(node_id) if tree is not None else None

        vc = self._visual_complexity(tree, node_id)
        idens = self._interaction_density(tree, node_id)
        depth = float(self._depth_in_tree(tree, node_id))
        n_ch = float(self._n_choices(tree, node_id))
        td = self._target_distance(tree, node_id)
        wm = self._working_memory_load(state)
        tp = self._task_progress(state)

        # Bounding-box area (normalised by viewport)
        bbox_area = 0.0
        if node is not None:
            bbox = getattr(node, "bounding_box", None)
            if bbox is not None:
                raw_area = getattr(bbox, "area", 0.0)
                viewport_area = self.viewport_width * self.viewport_height
                bbox_area = raw_area / max(viewport_area, 1.0)

        # Siblings
        n_siblings = 0.0
        if node is not None:
            parent_id = getattr(node, "parent_id", None)
            if parent_id and tree is not None:
                parent = tree.get_node(parent_id)
                if parent:
                    n_siblings = float(len(getattr(parent, "children", [])) - 1)

        # Subtree size
        subtree_sz = 1.0
        if node is not None:
            subtree_sz = float(getattr(node, "subtree_size", lambda: 1)())

        return np.array(
            [vc, idens, depth, n_ch, td, wm, tp, bbox_area, n_siblings, subtree_sz],
            dtype=np.float64,
        )

    @staticmethod
    def feature_names() -> list[str]:
        """Return the ordered list of feature names."""
        return list(_FEATURE_NAMES)

    # ── Individual feature extractors -------------------------------------

    def _visual_complexity(self, tree: Any, node_id: str) -> float:
        """Count visible elements in the viewport vicinity of *node_id*.

        Uses the focused node's bounding box to define a viewport region
        and counts all visible nodes whose bounding boxes overlap with it.
        Falls back to the global count of visible nodes when positional
        information is unavailable.
        """
        if tree is None:
            return 0.0

        node = tree.get_node(node_id)
        visible = tree.get_visible_nodes()

        if node is None or getattr(node, "bounding_box", None) is None:
            return float(len(visible))

        # Define viewport region centred on node
        bbox = node.bounding_box
        vp_x = max(0.0, bbox.center_x - self.viewport_width / 2)
        vp_y = max(0.0, bbox.center_y - self.viewport_height / 2)

        count = 0
        for vn in visible:
            vn_bbox = getattr(vn, "bounding_box", None)
            if vn_bbox is None:
                count += 1  # assume in viewport if no bbox
                continue
            # Check overlap with viewport
            if (
                vn_bbox.right >= vp_x
                and vn_bbox.x <= vp_x + self.viewport_width
                and vn_bbox.bottom >= vp_y
                and vn_bbox.y <= vp_y + self.viewport_height
            ):
                count += 1

        return float(count)

    def _interaction_density(self, tree: Any, node_id: str) -> float:
        """Fraction of sibling/nearby elements that are interactive.

        Computed as |interactive siblings| / |total siblings|.  If the node
        has no parent (root), returns the global interactive fraction.
        """
        if tree is None:
            return 0.0

        node = tree.get_node(node_id)
        if node is None:
            return 0.0

        parent_id = getattr(node, "parent_id", None)
        if parent_id is None:
            # Global density
            total = len(tree.node_index)
            interactive = len(tree.get_interactive_nodes())
            return interactive / max(total, 1)

        parent = tree.get_node(parent_id)
        if parent is None:
            return 0.0

        siblings = getattr(parent, "children", [])
        n_total = len(siblings)
        if n_total == 0:
            return 0.0

        n_interactive = sum(
            1 for s in siblings
            if getattr(s, "is_interactive", lambda: False)()
        )
        return n_interactive / n_total

    def _depth_in_tree(self, tree: Any, node_id: str) -> int:
        """Return the depth of *node_id* in the tree."""
        if tree is None:
            return 0
        node = tree.get_node(node_id)
        if node is None:
            return 0
        return int(getattr(node, "depth", 0))

    def _n_choices(self, tree: Any, node_id: str) -> int:
        """Number of interactive elements reachable from the current node.

        Approximated as the number of interactive children + interactive
        siblings.
        """
        if tree is None:
            return 0
        node = tree.get_node(node_id)
        if node is None:
            return 0

        count = 0
        # Interactive children
        for child in getattr(node, "children", []):
            if getattr(child, "is_interactive", lambda: False)():
                count += 1

        # Interactive siblings
        parent_id = getattr(node, "parent_id", None)
        if parent_id:
            parent = tree.get_node(parent_id)
            if parent:
                for sib in getattr(parent, "children", []):
                    if sib.id != node_id and getattr(
                        sib, "is_interactive", lambda: False
                    )():
                        count += 1

        return count

    def _target_distance(self, tree: Any, current_node_id: str) -> float:
        """Shortest tree-path distance from current to nearest target.

        Distance is measured in number of hops through the tree (not
        Euclidean distance).  Returns 0 if the current node is a target,
        or a large number if no path exists.
        """
        if not self.target_node_ids:
            return 0.0

        if current_node_id in self.target_node_ids:
            return 0.0

        if tree is None:
            return float(len(self.target_node_ids))

        # BFS from current node through parent/child links
        min_dist = float("inf")
        target_set = set(self.target_node_ids)

        visited: set[str] = set()
        queue: deque[tuple[str, int]] = deque([(current_node_id, 0)])

        while queue:
            nid, d = queue.popleft()
            if nid in visited:
                continue
            visited.add(nid)

            if nid in target_set:
                min_dist = min(min_dist, d)
                continue  # don't stop; there may be a closer target

            node = tree.get_node(nid)
            if node is None:
                continue

            # Expand children
            for child in getattr(node, "children", []):
                if child.id not in visited:
                    queue.append((child.id, d + 1))

            # Expand parent
            pid = getattr(node, "parent_id", None)
            if pid and pid not in visited:
                queue.append((pid, d + 1))

        return float(min_dist) if min_dist != float("inf") else 100.0

    def _working_memory_load(self, state: State) -> float:
        """Estimated working-memory load for *state*.

        Uses the ``working_memory_load`` feature if available, otherwise
        estimates from metadata (number of items held in WM).

        Miller's law: WM capacity ≈ 7 ± 2 chunks.
        """
        if "working_memory_load" in state.features:
            return state.features["working_memory_load"]

        wm_items = state.metadata.get("working_memory", [])
        return float(min(len(wm_items), 9))

    def _task_progress(self, state: State) -> float:
        """Fraction of task sub-goals completed."""
        return state.features.get("task_progress", 0.0)
