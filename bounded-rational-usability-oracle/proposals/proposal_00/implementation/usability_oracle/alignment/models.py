"""
usability_oracle.alignment.models — Data models for semantic tree alignment.

Defines the core data structures used across all three alignment passes:
exact matching, fuzzy bipartite matching, and residual classification.
Includes local definitions of accessibility-tree primitives so that the
alignment module is self-contained until ``usability_oracle.core`` is fully
implemented.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence


# ============================================================================
# Local accessibility-tree primitives
# ============================================================================

class AccessibilityRole(enum.Enum):
    """WAI-ARIA role taxonomy (subset used by the alignment module)."""

    # Landmark roles
    BANNER = "banner"
    COMPLEMENTARY = "complementary"
    CONTENT_INFO = "contentinfo"
    FORM = "form"
    MAIN = "main"
    NAVIGATION = "navigation"
    REGION = "region"
    SEARCH = "search"

    # Widget roles
    BUTTON = "button"
    CHECKBOX = "checkbox"
    COMBOBOX = "combobox"
    DIALOG = "dialog"
    GRID = "grid"
    GRID_CELL = "gridcell"
    LINK = "link"
    LIST = "list"
    LIST_ITEM = "listitem"
    MENU = "menu"
    MENU_ITEM = "menuitem"
    OPTION = "option"
    RADIO = "radio"
    SCROLLBAR = "scrollbar"
    SLIDER = "slider"
    SPINBUTTON = "spinbutton"
    TAB = "tab"
    TAB_LIST = "tablist"
    TAB_PANEL = "tabpanel"
    TEXTBOX = "textbox"
    TREE = "tree"
    TREE_ITEM = "treeitem"

    # Document-structure roles
    ARTICLE = "article"
    CELL = "cell"
    COLUMN_HEADER = "columnheader"
    DEFINITION = "definition"
    DOCUMENT = "document"
    GROUP = "group"
    HEADING = "heading"
    IMG = "img"
    LIST_BOX = "listbox"
    LOG = "log"
    MARQUEE = "marquee"
    MATH = "math"
    NOTE = "note"
    PARAGRAPH = "paragraph"
    PRESENTATION = "presentation"
    ROW = "row"
    ROW_GROUP = "rowgroup"
    ROW_HEADER = "rowheader"
    SEPARATOR = "separator"
    STATUS = "status"
    TABLE = "table"
    TERM = "term"
    TIMER = "timer"
    TOOLBAR = "toolbar"
    TOOLTIP = "tooltip"

    # Generic
    GENERIC = "generic"
    NONE = "none"
    UNKNOWN = "unknown"


# Role taxonomy groups for distance computation
_ROLE_GROUPS: dict[str, set[AccessibilityRole]] = {
    "landmark": {
        AccessibilityRole.BANNER, AccessibilityRole.COMPLEMENTARY,
        AccessibilityRole.CONTENT_INFO, AccessibilityRole.FORM,
        AccessibilityRole.MAIN, AccessibilityRole.NAVIGATION,
        AccessibilityRole.REGION, AccessibilityRole.SEARCH,
    },
    "widget": {
        AccessibilityRole.BUTTON, AccessibilityRole.CHECKBOX,
        AccessibilityRole.COMBOBOX, AccessibilityRole.DIALOG,
        AccessibilityRole.LINK, AccessibilityRole.MENU,
        AccessibilityRole.MENU_ITEM, AccessibilityRole.RADIO,
        AccessibilityRole.SLIDER, AccessibilityRole.SPINBUTTON,
        AccessibilityRole.TAB, AccessibilityRole.TEXTBOX,
    },
    "list": {
        AccessibilityRole.LIST, AccessibilityRole.LIST_ITEM,
        AccessibilityRole.LIST_BOX, AccessibilityRole.OPTION,
    },
    "table": {
        AccessibilityRole.TABLE, AccessibilityRole.GRID,
        AccessibilityRole.CELL, AccessibilityRole.GRID_CELL,
        AccessibilityRole.ROW, AccessibilityRole.ROW_GROUP,
        AccessibilityRole.ROW_HEADER, AccessibilityRole.COLUMN_HEADER,
    },
    "navigation": {
        AccessibilityRole.MENU, AccessibilityRole.MENU_ITEM,
        AccessibilityRole.TAB, AccessibilityRole.TAB_LIST,
        AccessibilityRole.TAB_PANEL, AccessibilityRole.TREE,
        AccessibilityRole.TREE_ITEM,
    },
    "structure": {
        AccessibilityRole.ARTICLE, AccessibilityRole.DOCUMENT,
        AccessibilityRole.GROUP, AccessibilityRole.HEADING,
        AccessibilityRole.PARAGRAPH, AccessibilityRole.SEPARATOR,
    },
}


def role_taxonomy_distance(a: AccessibilityRole, b: AccessibilityRole) -> float:
    """Return a normalised [0, 1] distance between two roles.

    * 0.0 — identical roles.
    * 0.3 — roles in the same taxonomy group.
    * 0.7 — roles in different groups but both non-generic.
    * 1.0 — one or both roles are GENERIC / NONE / UNKNOWN.
    """
    if a == b:
        return 0.0
    sentinel = {AccessibilityRole.GENERIC, AccessibilityRole.NONE, AccessibilityRole.UNKNOWN}
    if a in sentinel or b in sentinel:
        return 1.0
    for _group_name, members in _ROLE_GROUPS.items():
        if a in members and b in members:
            return 0.3
    return 0.7


@dataclass(frozen=True)
class Point2D:
    """A 2-D point in screen-pixel coordinates."""

    x: float
    y: float

    def distance_to(self, other: Point2D) -> float:
        return math.hypot(self.x - other.x, self.y - other.y)


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned bounding rectangle in screen pixels."""

    x: float
    y: float
    width: float
    height: float

    @property
    def center(self) -> Point2D:
        return Point2D(self.x + self.width / 2.0, self.y + self.height / 2.0)

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    def iou(self, other: BoundingBox) -> float:
        """Intersection-over-union of two bounding boxes."""
        x1 = max(self.x, other.x)
        y1 = max(self.y, other.y)
        x2 = min(self.x + self.width, other.x + other.width)
        y2 = min(self.y + self.height, other.y + other.height)
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0


@dataclass
class AccessibilityNode:
    """A single node in an accessibility tree.

    Carries the semantics needed for alignment: role, name, bounding box,
    properties, and structural information (parent/children).
    """

    node_id: str
    role: AccessibilityRole
    name: str = ""
    description: str = ""
    value: str = ""
    bounding_box: Optional[BoundingBox] = None
    properties: dict[str, Any] = field(default_factory=dict)
    parent_id: Optional[str] = None
    children_ids: list[str] = field(default_factory=list)
    depth: int = 0
    tree_path: str = ""

    @property
    def is_leaf(self) -> bool:
        return len(self.children_ids) == 0

    @property
    def subtree_size(self) -> int:
        """Approximate subtree size stored in properties, or 1 for leaves."""
        return int(self.properties.get("subtree_size", 1))


@dataclass
class AccessibilityTree:
    """An ordered forest of :class:`AccessibilityNode` objects.

    Maintains a flat dict ``nodes`` keyed by *node_id* together with a list of
    ``root_ids`` that records top-level ordering.
    """

    nodes: dict[str, AccessibilityNode] = field(default_factory=dict)
    root_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_node(self, node_id: str) -> AccessibilityNode:
        return self.nodes[node_id]

    def get_children(self, node_id: str) -> list[AccessibilityNode]:
        node = self.nodes[node_id]
        return [self.nodes[cid] for cid in node.children_ids if cid in self.nodes]

    def get_parent(self, node_id: str) -> Optional[AccessibilityNode]:
        pid = self.nodes[node_id].parent_id
        return self.nodes.get(pid) if pid else None

    def all_node_ids(self) -> list[str]:
        return list(self.nodes.keys())

    def leaf_ids(self) -> list[str]:
        return [nid for nid, n in self.nodes.items() if n.is_leaf]

    def node_count(self) -> int:
        return len(self.nodes)

    # ------------------------------------------------------------------
    # Tree-path helpers
    # ------------------------------------------------------------------

    def compute_paths(self) -> None:
        """Populate ``tree_path`` and ``depth`` for every node via BFS."""
        visited: set[str] = set()
        queue: list[tuple[str, str, int]] = [(rid, f"/{rid}", 0) for rid in self.root_ids]
        while queue:
            nid, path, depth = queue.pop(0)
            if nid in visited or nid not in self.nodes:
                continue
            visited.add(nid)
            node = self.nodes[nid]
            node.tree_path = path
            node.depth = depth
            for cid in node.children_ids:
                queue.append((cid, f"{path}/{cid}", depth + 1))

    def compute_subtree_sizes(self) -> None:
        """Bottom-up computation of subtree sizes."""
        order = self._postorder()
        for nid in order:
            node = self.nodes[nid]
            children = self.get_children(nid)
            node.properties["subtree_size"] = 1 + sum(
                c.properties.get("subtree_size", 1) for c in children
            )

    def _postorder(self) -> list[str]:
        """Return node-ids in post-order (children before parents)."""
        result: list[str] = []
        visited: set[str] = set()

        def _visit(nid: str) -> None:
            if nid in visited or nid not in self.nodes:
                return
            visited.add(nid)
            for cid in self.nodes[nid].children_ids:
                _visit(cid)
            result.append(nid)

        for rid in self.root_ids:
            _visit(rid)
        return result

    def subtree_node_ids(self, root_id: str) -> list[str]:
        """Return all node-ids in the subtree rooted at *root_id*."""
        result: list[str] = []
        stack = [root_id]
        while stack:
            nid = stack.pop()
            if nid not in self.nodes:
                continue
            result.append(nid)
            stack.extend(reversed(self.nodes[nid].children_ids))
        return result


# ============================================================================
# Alignment-specific enumerations
# ============================================================================

class AlignmentPass(enum.Enum):
    """Which alignment pass produced a particular mapping."""

    EXACT_HASH = "exact_hash"
    EXACT_ID = "exact_id"
    EXACT_PATH = "exact_path"
    FUZZY = "fuzzy"
    RESIDUAL = "residual"


class EditOperationType(enum.Enum):
    """Kinds of edit operations in the semantic diff."""

    ADD = "add"
    REMOVE = "remove"
    RENAME = "rename"
    RETYPE = "retype"
    MOVE = "move"
    RESIZE = "resize"
    REORDER = "reorder"
    MODIFY_PROPERTY = "modify_property"
    RESTRUCTURE = "restructure"


# ============================================================================
# Alignment data-classes
# ============================================================================

@dataclass(frozen=True)
class EditOperation:
    """A single edit operation detected between two tree versions."""

    operation_type: EditOperationType
    source_node_id: Optional[str]
    target_node_id: Optional[str]
    cost: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        src = self.source_node_id or "∅"
        tgt = self.target_node_id or "∅"
        return f"{self.operation_type.value}: {src} → {tgt} (cost={self.cost:.3f})"


@dataclass(frozen=True)
class NodeMapping:
    """A pairing between a source-tree node and a target-tree node."""

    source_id: str
    target_id: str
    confidence: float = 1.0
    pass_matched: AlignmentPass = AlignmentPass.EXACT_HASH

    def __str__(self) -> str:
        return (
            f"{self.source_id} ↔ {self.target_id} "
            f"[{self.pass_matched.value}, conf={self.confidence:.2f}]"
        )


@dataclass
class AlignmentResult:
    """Aggregated result of the 3-pass alignment pipeline.

    Bundles the node mappings, detected edit operations, classified
    additions / removals, scalar metrics, and per-pass statistics.
    """

    mappings: list[NodeMapping] = field(default_factory=list)
    edit_operations: list[EditOperation] = field(default_factory=list)
    additions: list[str] = field(default_factory=list)
    removals: list[str] = field(default_factory=list)
    edit_distance: float = 0.0
    similarity_score: float = 1.0
    pass_statistics: dict[AlignmentPass, int] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_mapped_node(self, source_id: str) -> Optional[str]:
        """Return the target-id mapped to *source_id*, or ``None``."""
        for m in self.mappings:
            if m.source_id == source_id:
                return m.target_id
        return None

    def get_reverse_mapped_node(self, target_id: str) -> Optional[str]:
        """Return the source-id mapped to *target_id*, or ``None``."""
        for m in self.mappings:
            if m.target_id == target_id:
                return m.source_id
        return None

    def get_unmapped_source(self, source_tree: AccessibilityTree) -> list[str]:
        """Return source node-ids that were *not* paired with any target node."""
        matched = {m.source_id for m in self.mappings}
        return [nid for nid in source_tree.all_node_ids() if nid not in matched]

    def get_unmapped_target(self, target_tree: AccessibilityTree) -> list[str]:
        """Return target node-ids that were *not* paired with any source node."""
        matched = {m.target_id for m in self.mappings}
        return [nid for nid in target_tree.all_node_ids() if nid not in matched]

    def mappings_by_pass(self, ap: AlignmentPass) -> list[NodeMapping]:
        """Filter mappings to those produced by a particular pass."""
        return [m for m in self.mappings if m.pass_matched == ap]

    def average_confidence(self) -> float:
        """Mean confidence across all mappings."""
        if not self.mappings:
            return 0.0
        return sum(m.confidence for m in self.mappings) / len(self.mappings)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Human-readable summary string."""
        lines = [
            "Alignment Result",
            "=" * 50,
            f"  Mappings:          {len(self.mappings)}",
            f"  Edit operations:   {len(self.edit_operations)}",
            f"  Additions:         {len(self.additions)}",
            f"  Removals:          {len(self.removals)}",
            f"  Edit distance:     {self.edit_distance:.4f}",
            f"  Similarity score:  {self.similarity_score:.4f}",
            f"  Avg confidence:    {self.average_confidence():.4f}",
            "",
            "Pass statistics:",
        ]
        for ap in AlignmentPass:
            count = self.pass_statistics.get(ap, 0)
            lines.append(f"  {ap.value:15s}  {count}")
        lines.append("")
        if self.edit_operations:
            lines.append("Edit operations:")
            for op in self.edit_operations[:20]:
                lines.append(f"  {op}")
            if len(self.edit_operations) > 20:
                lines.append(f"  ... and {len(self.edit_operations) - 20} more")
        return "\n".join(lines)


@dataclass
class AlignmentConfig:
    """Configuration knobs for the alignment pipeline."""

    # Pass-1 exact matching
    enable_hash_match: bool = True
    enable_id_match: bool = True
    enable_path_match: bool = True

    # Pass-2 fuzzy matching
    fuzzy_threshold: float = 0.40
    role_weight: float = 0.30
    name_weight: float = 0.30
    position_weight: float = 0.20
    structure_weight: float = 0.20
    position_sigma: float = 50.0

    # Pass-3 residual classification
    move_threshold: float = 0.80
    rename_threshold: float = 0.70
    retype_threshold: float = 0.60

    # Cost model
    add_cost_per_node: float = 1.0
    remove_cost_per_node: float = 1.0
    rename_base_cost: float = 0.5
    retype_base_cost: float = 0.8
    move_base_cost: float = 0.6
    resize_base_cost: float = 0.3

    # Cognitive weighting
    cognitive_weight_interactive: float = 2.0
    cognitive_weight_landmark: float = 1.5
    cognitive_weight_structure: float = 1.0

    def validate(self) -> list[str]:
        """Return a list of validation error messages (empty ⇒ OK)."""
        errors: list[str] = []
        weights = [self.role_weight, self.name_weight, self.position_weight, self.structure_weight]
        if any(w < 0 for w in weights):
            errors.append("Similarity weights must be non-negative.")
        total = sum(weights)
        if abs(total - 1.0) > 1e-6:
            errors.append(f"Similarity weights must sum to 1.0, got {total:.6f}.")
        if not (0.0 <= self.fuzzy_threshold <= 1.0):
            errors.append("fuzzy_threshold must be in [0, 1].")
        return errors


@dataclass
class AlignmentContext:
    """Runtime context passed through the three alignment passes."""

    source_tree: AccessibilityTree
    target_tree: AccessibilityTree
    config: AlignmentConfig = field(default_factory=AlignmentConfig)
    _cache: dict[str, Any] = field(default_factory=dict, repr=False)

    def cache_get(self, key: str) -> Any:
        return self._cache.get(key)

    def cache_set(self, key: str, value: Any) -> None:
        self._cache[key] = value
