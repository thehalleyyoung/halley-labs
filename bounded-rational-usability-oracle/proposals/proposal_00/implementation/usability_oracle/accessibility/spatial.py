"""Spatial layout analysis and Fitts' law computations for accessibility nodes."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from usability_oracle.accessibility.models import (
    AccessibilityNode,
    AccessibilityTree,
    BoundingBox,
)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class LayoutInfo:
    """Aggregate layout metrics for a set of nodes."""

    total_nodes: int = 0
    visible_nodes: int = 0
    viewport: Optional[BoundingBox] = None
    visual_density: float = 0.0
    spacing_regularity: float = 0.0
    mean_element_area: float = 0.0
    reading_order: list[str] = field(default_factory=list)
    groups: list[NodeGroup] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_nodes": self.total_nodes,
            "visible_nodes": self.visible_nodes,
            "viewport": self.viewport.to_dict() if self.viewport else None,
            "visual_density": self.visual_density,
            "spacing_regularity": self.spacing_regularity,
            "mean_element_area": self.mean_element_area,
            "reading_order": self.reading_order,
            "groups": [g.to_dict() for g in self.groups],
        }


@dataclass
class NodeGroup:
    """A cluster of spatially proximate nodes."""

    group_id: int
    node_ids: list[str]
    bounding_box: BoundingBox
    centroid_x: float = 0.0
    centroid_y: float = 0.0
    density: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "node_ids": self.node_ids,
            "bounding_box": self.bounding_box.to_dict(),
            "centroid_x": self.centroid_x,
            "centroid_y": self.centroid_y,
            "density": self.density,
        }


# ── Spatial analyzer ──────────────────────────────────────────────────────────

class SpatialAnalyzer:
    """Spatial analysis utilities for accessibility trees."""

    def __init__(self, default_viewport: Optional[BoundingBox] = None) -> None:
        self._default_viewport = default_viewport or BoundingBox(0, 0, 1920, 1080)

    # ── High-level API ────────────────────────────────────────────────────

    def compute_layout(self, tree: AccessibilityTree) -> LayoutInfo:
        """Compute aggregate layout information for the entire tree."""
        all_nodes = list(tree.node_index.values())
        visible = [n for n in all_nodes if n.is_visible() and n.bounding_box is not None]

        viewport = self._compute_viewport(visible)
        groups = self.detect_groups(visible) if len(visible) >= 2 else []
        reading = self.compute_reading_order(visible)

        areas = [n.bounding_box.area for n in visible if n.bounding_box is not None]
        mean_area = float(np.mean(areas)) if areas else 0.0

        return LayoutInfo(
            total_nodes=len(all_nodes),
            visible_nodes=len(visible),
            viewport=viewport,
            visual_density=self.compute_visual_density(viewport, visible),
            spacing_regularity=self.compute_spacing_regularity(visible),
            mean_element_area=mean_area,
            reading_order=[n.id for n in reading],
            groups=groups,
        )

    def _compute_viewport(self, nodes: list[AccessibilityNode]) -> BoundingBox:
        """Determine viewport from node extents or use default."""
        boxes = [n.bounding_box for n in nodes if n.bounding_box is not None]
        if not boxes:
            return self._default_viewport
        min_x = min(b.x for b in boxes)
        min_y = min(b.y for b in boxes)
        max_r = max(b.right for b in boxes)
        max_b = max(b.bottom for b in boxes)
        return BoundingBox(min_x, min_y, max_r - min_x, max_b - min_y)

    # ── Grouping via hierarchical clustering ──────────────────────────────

    def detect_groups(
        self,
        nodes: list[AccessibilityNode],
        distance_threshold: float = 100.0,
    ) -> list[NodeGroup]:
        """Cluster nodes by spatial proximity using agglomerative clustering.

        Uses scipy's hierarchy module with average linkage and a distance
        threshold to cut the dendrogram.
        """
        boxes = [n for n in nodes if n.bounding_box is not None]
        if len(boxes) < 2:
            return []

        centers = np.array(
            [[n.bounding_box.center_x, n.bounding_box.center_y] for n in boxes]
        )

        try:
            from scipy.cluster.hierarchy import fcluster, linkage
            from scipy.spatial.distance import pdist

            dists = pdist(centers, metric="euclidean")
            Z = linkage(dists, method="average")
            labels = fcluster(Z, t=distance_threshold, criterion="distance")
        except ImportError:
            # Fallback: simple grid-based grouping when scipy is unavailable
            labels = self._grid_cluster(centers, distance_threshold)

        groups_dict: dict[int, list[int]] = {}
        for idx, label in enumerate(labels):
            groups_dict.setdefault(int(label), []).append(idx)

        result: list[NodeGroup] = []
        for gid, indices in sorted(groups_dict.items()):
            group_nodes = [boxes[i] for i in indices]
            node_ids = [n.id for n in group_nodes]
            group_boxes = [n.bounding_box for n in group_nodes if n.bounding_box is not None]

            bbox = group_boxes[0]
            for b in group_boxes[1:]:
                bbox = bbox.union(b)

            cx = float(np.mean([b.center_x for b in group_boxes]))
            cy = float(np.mean([b.center_y for b in group_boxes]))
            density = sum(b.area for b in group_boxes) / max(bbox.area, 1e-9)

            result.append(
                NodeGroup(
                    group_id=gid,
                    node_ids=node_ids,
                    bounding_box=bbox,
                    centroid_x=cx,
                    centroid_y=cy,
                    density=density,
                )
            )
        return result

    @staticmethod
    def _grid_cluster(centers: np.ndarray, threshold: float) -> list[int]:
        """Simple fallback clustering: assign to grid cells."""
        if len(centers) == 0:
            return []
        cell_size = max(threshold, 1.0)
        labels: list[int] = []
        cell_map: dict[tuple[int, int], int] = {}
        next_id = 1
        for cx, cy in centers:
            cell = (int(cx // cell_size), int(cy // cell_size))
            if cell not in cell_map:
                cell_map[cell] = next_id
                next_id += 1
            labels.append(cell_map[cell])
        return labels

    # ── Visual density ────────────────────────────────────────────────────

    def compute_visual_density(
        self,
        region: BoundingBox,
        nodes: list[AccessibilityNode],
    ) -> float:
        """Fraction of region area covered by node bounding boxes.

        Uses a sweep-based approximation: for each node whose bbox overlaps
        the region, accumulate the intersection area, then divide by region
        area. Overlapping elements may push density above 1.0.
        """
        if region.area <= 0:
            return 0.0
        total_cover = 0.0
        for node in nodes:
            if node.bounding_box is None:
                continue
            inter = region.intersection(node.bounding_box)
            if inter is not None:
                total_cover += inter.area
        return total_cover / region.area

    # ── Spacing regularity ────────────────────────────────────────────────

    def compute_spacing_regularity(self, nodes: list[AccessibilityNode]) -> float:
        """Measure how regular the spacing is between adjacent elements.

        Returns a value in [0, 1]. 1.0 means perfectly regular spacing;
        lower values indicate irregular gaps. Based on coefficient of
        variation of nearest-neighbor distances.
        """
        boxes = [n for n in nodes if n.bounding_box is not None]
        if len(boxes) < 2:
            return 1.0

        centers = np.array(
            [[n.bounding_box.center_x, n.bounding_box.center_y] for n in boxes]
        )

        # Compute nearest-neighbor distances
        nn_dists: list[float] = []
        for i in range(len(centers)):
            best = float("inf")
            for j in range(len(centers)):
                if i == j:
                    continue
                d = float(np.linalg.norm(centers[i] - centers[j]))
                if d < best:
                    best = d
            if best < float("inf"):
                nn_dists.append(best)

        if not nn_dists:
            return 1.0

        mean_d = float(np.mean(nn_dists))
        std_d = float(np.std(nn_dists))
        if mean_d == 0:
            return 1.0

        cv = std_d / mean_d  # coefficient of variation
        return 1.0 / (1.0 + cv)

    # ── Nearest neighbours ────────────────────────────────────────────────

    def nearest_neighbors(
        self,
        node: AccessibilityNode,
        candidates: list[AccessibilityNode],
        k: int = 5,
    ) -> list[tuple[AccessibilityNode, float]]:
        """Return k nearest neighbours of *node* sorted by distance.

        Distance is center-to-center Euclidean distance on bounding boxes.
        Nodes without bounding boxes are skipped.
        """
        if node.bounding_box is None:
            return []

        dists: list[tuple[AccessibilityNode, float]] = []
        for c in candidates:
            if c.id == node.id or c.bounding_box is None:
                continue
            d = node.bounding_box.distance_to(c.bounding_box)
            dists.append((c, d))

        dists.sort(key=lambda x: x[1])
        return dists[:k]

    # ── Fitts' law helpers ────────────────────────────────────────────────

    @staticmethod
    def fitts_distance(source: BoundingBox, target: BoundingBox) -> float:
        """Center-to-center Euclidean distance between two boxes."""
        return source.distance_to(target)

    @staticmethod
    def fitts_width(target: BoundingBox, approach_angle: float = 0.0) -> float:
        """Effective target width along the approach angle (radians).

        For a rectangle, the width that the user "sees" when approaching
        from angle θ is: W_eff = W·|cos θ| + H·|sin θ| (projected width
        along the perpendicular of the approach direction).
        """
        w = target.width
        h = target.height
        return abs(w * math.cos(approach_angle)) + abs(h * math.sin(approach_angle))

    @staticmethod
    def fitts_index_of_difficulty(distance: float, width: float) -> float:
        """Shannon formulation: ID = log2(D / W + 1)."""
        if width <= 0:
            return float("inf")
        return math.log2(distance / width + 1.0)

    def fitts_id(self, source: BoundingBox, target: BoundingBox) -> float:
        """Compute Fitts' index of difficulty between two bounding boxes."""
        d = self.fitts_distance(source, target)
        angle = math.atan2(
            target.center_y - source.center_y,
            target.center_x - source.center_x,
        )
        w = self.fitts_width(target, angle)
        return self.fitts_index_of_difficulty(d, w)

    # ── Reading order ─────────────────────────────────────────────────────

    def compute_reading_order(
        self,
        nodes: list[AccessibilityNode],
        rtl: bool = False,
    ) -> list[AccessibilityNode]:
        """Sort nodes into visual reading order (top-to-bottom, left-to-right).

        Groups nodes into horizontal bands and sorts within each band.
        """
        positioned = [n for n in nodes if n.bounding_box is not None]
        if not positioned:
            return list(nodes)

        # Sort by vertical position
        positioned.sort(key=lambda n: n.bounding_box.center_y)  # type: ignore[union-attr]

        # Group into lines: nodes whose vertical centers are within half
        # of the median element height of each other
        heights = [n.bounding_box.height for n in positioned if n.bounding_box is not None]
        band_threshold = float(np.median(heights)) * 0.6 if heights else 20.0

        lines: list[list[AccessibilityNode]] = []
        current_line: list[AccessibilityNode] = [positioned[0]]
        current_y = positioned[0].bounding_box.center_y  # type: ignore[union-attr]

        for node in positioned[1:]:
            cy = node.bounding_box.center_y  # type: ignore[union-attr]
            if abs(cy - current_y) <= band_threshold:
                current_line.append(node)
            else:
                lines.append(current_line)
                current_line = [node]
                current_y = cy
        lines.append(current_line)

        # Sort within each line by horizontal position
        result: list[AccessibilityNode] = []
        for line in lines:
            line.sort(
                key=lambda n: n.bounding_box.center_x,  # type: ignore[union-attr]
                reverse=rtl,
            )
            result.extend(line)

        return result

    # ── Overlap analysis ──────────────────────────────────────────────────

    def find_overlapping_pairs(
        self,
        nodes: list[AccessibilityNode],
    ) -> list[tuple[str, str, float]]:
        """Return pairs of overlapping nodes with overlap area."""
        pairs: list[tuple[str, str, float]] = []
        positioned = [n for n in nodes if n.bounding_box is not None]
        for i in range(len(positioned)):
            for j in range(i + 1, len(positioned)):
                a = positioned[i].bounding_box
                b = positioned[j].bounding_box
                if a is not None and b is not None:
                    inter = a.intersection(b)
                    if inter is not None and inter.area > 0:
                        pairs.append((positioned[i].id, positioned[j].id, inter.area))
        return pairs

    def compute_alignment_score(self, nodes: list[AccessibilityNode]) -> float:
        """Measure how well nodes are aligned on common edges.

        Returns [0, 1] where 1.0 means elements share common x or y edges.
        """
        positioned = [n for n in nodes if n.bounding_box is not None]
        if len(positioned) < 2:
            return 1.0

        left_edges = [n.bounding_box.x for n in positioned if n.bounding_box]
        top_edges = [n.bounding_box.y for n in positioned if n.bounding_box]

        def _alignment_for_values(values: list[float]) -> float:
            if len(values) < 2:
                return 1.0
            arr = np.array(sorted(values))
            # Count how many values share (approximately) the same coordinate
            tolerance = 2.0  # pixels
            aligned = 0
            total = len(arr) - 1
            for i in range(total):
                if abs(arr[i + 1] - arr[i]) <= tolerance:
                    aligned += 1
            return aligned / max(total, 1)

        align_x = _alignment_for_values(left_edges)
        align_y = _alignment_for_values(top_edges)
        return (align_x + align_y) / 2.0
