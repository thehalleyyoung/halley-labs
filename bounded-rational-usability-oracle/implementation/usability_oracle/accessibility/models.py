"""Core accessibility tree data models with traversal algorithms."""

from __future__ import annotations

import hashlib
import json
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

from usability_oracle.accessibility.roles import RoleTaxonomy


# ── Bounding box ──────────────────────────────────────────────────────────────

@dataclass
class BoundingBox:
    """Axis-aligned bounding box in screen coordinates."""

    x: float
    y: float
    width: float
    height: float

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def bottom(self) -> float:
        return self.y + self.height

    @property
    def center_x(self) -> float:
        return self.x + self.width / 2.0

    @property
    def center_y(self) -> float:
        return self.y + self.height / 2.0

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    def contains_point(self, px: float, py: float) -> bool:
        return self.x <= px <= self.right and self.y <= py <= self.bottom

    def contains(self, other: BoundingBox) -> bool:
        return (
            self.x <= other.x
            and self.y <= other.y
            and self.right >= other.right
            and self.bottom >= other.bottom
        )

    def overlaps(self, other: BoundingBox) -> bool:
        return not (
            self.right < other.x
            or other.right < self.x
            or self.bottom < other.y
            or other.bottom < self.y
        )

    def intersection(self, other: BoundingBox) -> Optional[BoundingBox]:
        ix = max(self.x, other.x)
        iy = max(self.y, other.y)
        ir = min(self.right, other.right)
        ib = min(self.bottom, other.bottom)
        if ir > ix and ib > iy:
            return BoundingBox(ix, iy, ir - ix, ib - iy)
        return None

    def union(self, other: BoundingBox) -> BoundingBox:
        ux = min(self.x, other.x)
        uy = min(self.y, other.y)
        ur = max(self.right, other.right)
        ub = max(self.bottom, other.bottom)
        return BoundingBox(ux, uy, ur - ux, ub - uy)

    def distance_to(self, other: BoundingBox) -> float:
        """Euclidean center-to-center distance."""
        dx = self.center_x - other.center_x
        dy = self.center_y - other.center_y
        return (dx * dx + dy * dy) ** 0.5

    def to_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BoundingBox:
        return cls(
            x=float(data["x"]),
            y=float(data["y"]),
            width=float(data["width"]),
            height=float(data["height"]),
        )


# ── Accessibility property ────────────────────────────────────────────────────

@dataclass
class AccessibilityProperty:
    """A single named property on an accessibility node."""

    name: str
    value: Any

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "value": self.value}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AccessibilityProperty:
        return cls(name=data["name"], value=data["value"])


# ── Accessibility state ──────────────────────────────────────────────────────

@dataclass
class AccessibilityState:
    """Captures the interactive state of an accessibility node."""

    focused: bool = False
    selected: bool = False
    expanded: bool = False
    checked: Optional[bool] = None
    disabled: bool = False
    hidden: bool = False
    required: bool = False
    readonly: bool = False
    pressed: Optional[bool] = None
    value: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "focused": self.focused,
            "selected": self.selected,
            "expanded": self.expanded,
            "disabled": self.disabled,
            "hidden": self.hidden,
            "required": self.required,
            "readonly": self.readonly,
        }
        if self.checked is not None:
            d["checked"] = self.checked
        if self.pressed is not None:
            d["pressed"] = self.pressed
        if self.value is not None:
            d["value"] = self.value
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AccessibilityState:
        return cls(
            focused=data.get("focused", False),
            selected=data.get("selected", False),
            expanded=data.get("expanded", False),
            checked=data.get("checked"),
            disabled=data.get("disabled", False),
            hidden=data.get("hidden", False),
            required=data.get("required", False),
            readonly=data.get("readonly", False),
            pressed=data.get("pressed"),
            value=data.get("value"),
        )


# ── Accessibility node ───────────────────────────────────────────────────────

_taxonomy = RoleTaxonomy()


@dataclass
class AccessibilityNode:
    """A single node in the accessibility tree."""

    id: str
    role: str
    name: str
    description: str = ""
    bounding_box: Optional[BoundingBox] = None
    properties: dict[str, Any] = field(default_factory=dict)
    state: AccessibilityState = field(default_factory=AccessibilityState)
    children: list[AccessibilityNode] = field(default_factory=list)
    parent_id: Optional[str] = None
    depth: int = 0
    index_in_parent: int = 0

    # ── Queries ───────────────────────────────────────────────────────────

    def is_interactive(self) -> bool:
        """True if this node has an interactive ARIA role."""
        return _taxonomy.is_interactive(self.role)

    def is_visible(self) -> bool:
        """True if the node is not hidden."""
        return not self.state.hidden

    def is_focusable(self) -> bool:
        """True if the node can receive keyboard focus."""
        if self.state.disabled or self.state.hidden:
            return False
        if self.is_interactive():
            return True
        tabindex = self.properties.get("tabindex")
        if tabindex is not None:
            try:
                return int(tabindex) >= 0
            except (ValueError, TypeError):
                return False
        return False

    def semantic_hash(self) -> str:
        """Content-based hash of role, name and children structure."""
        h = hashlib.sha256()
        h.update(self.role.encode("utf-8"))
        h.update(self.name.encode("utf-8"))
        for child in self.children:
            h.update(child.semantic_hash().encode("utf-8"))
        return h.hexdigest()[:16]

    def subtree_size(self) -> int:
        """Number of nodes in the subtree rooted at this node."""
        count = 1
        stack = list(self.children)
        while stack:
            node = stack.pop()
            count += 1
            stack.extend(node.children)
        return count

    # ── Traversal helpers ─────────────────────────────────────────────────

    def get_ancestors(self, tree: AccessibilityTree) -> list[AccessibilityNode]:
        """Walk up parent_id links and return list from parent to root."""
        ancestors: list[AccessibilityNode] = []
        current_id = self.parent_id
        visited: set[str] = set()
        while current_id is not None and current_id not in visited:
            visited.add(current_id)
            node = tree.get_node(current_id)
            if node is None:
                break
            ancestors.append(node)
            current_id = node.parent_id
        return ancestors

    def get_descendants(self) -> list[AccessibilityNode]:
        """BFS list of all descendant nodes."""
        result: list[AccessibilityNode] = []
        queue: deque[AccessibilityNode] = deque(self.children)
        while queue:
            node = queue.popleft()
            result.append(node)
            queue.extend(node.children)
        return result

    def find_by_role(self, role: str) -> list[AccessibilityNode]:
        """Return all descendants (and self) matching a given role."""
        matches: list[AccessibilityNode] = []
        stack: list[AccessibilityNode] = [self]
        while stack:
            node = stack.pop()
            if node.role == role:
                matches.append(node)
            stack.extend(reversed(node.children))
        return matches

    def find_by_name(self, name: str, case_sensitive: bool = False) -> list[AccessibilityNode]:
        """Return all descendants (and self) whose name matches."""
        target = name if case_sensitive else name.lower()
        matches: list[AccessibilityNode] = []
        stack: list[AccessibilityNode] = [self]
        while stack:
            node = stack.pop()
            node_name = node.name if case_sensitive else node.name.lower()
            if node_name == target:
                matches.append(node)
            stack.extend(reversed(node.children))
        return matches

    def iter_preorder(self) -> Iterator[AccessibilityNode]:
        """Pre-order depth-first iteration."""
        stack: list[AccessibilityNode] = [self]
        while stack:
            node = stack.pop()
            yield node
            stack.extend(reversed(node.children))

    def iter_postorder(self) -> Iterator[AccessibilityNode]:
        """Post-order depth-first iteration."""
        stack: list[tuple[AccessibilityNode, bool]] = [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                yield node
            else:
                stack.append((node, True))
                for child in reversed(node.children):
                    stack.append((child, False))

    # ── Serialisation ─────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "role": self.role,
            "name": self.name,
            "description": self.description,
            "state": self.state.to_dict(),
            "properties": self.properties,
            "depth": self.depth,
            "index_in_parent": self.index_in_parent,
            "children": [c.to_dict() for c in self.children],
        }
        if self.parent_id is not None:
            d["parent_id"] = self.parent_id
        if self.bounding_box is not None:
            d["bounding_box"] = self.bounding_box.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AccessibilityNode:
        bbox = None
        if "bounding_box" in data and data["bounding_box"] is not None:
            bbox = BoundingBox.from_dict(data["bounding_box"])
        children = [cls.from_dict(c) for c in data.get("children", [])]
        return cls(
            id=data["id"],
            role=data["role"],
            name=data.get("name", ""),
            description=data.get("description", ""),
            bounding_box=bbox,
            properties=data.get("properties", {}),
            state=AccessibilityState.from_dict(data.get("state", {})),
            children=children,
            parent_id=data.get("parent_id"),
            depth=data.get("depth", 0),
            index_in_parent=data.get("index_in_parent", 0),
        )

    def __repr__(self) -> str:
        return (
            f"AccessibilityNode(id={self.id!r}, role={self.role!r}, "
            f"name={self.name!r}, children={len(self.children)})"
        )


# ── Accessibility tree ───────────────────────────────────────────────────────

@dataclass
class AccessibilityTree:
    """Rooted tree of accessibility nodes with index and algorithms."""

    root: AccessibilityNode
    metadata: dict[str, Any] = field(default_factory=dict)
    node_index: dict[str, AccessibilityNode] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.node_index:
            self.build_index()

    # ── Index management ──────────────────────────────────────────────────

    def build_index(self) -> None:
        """Walk the tree and build id -> node map, fixing parent_id links."""
        self.node_index.clear()
        stack: list[tuple[AccessibilityNode, Optional[str]]] = [(self.root, None)]
        while stack:
            node, pid = stack.pop()
            node.parent_id = pid
            self.node_index[node.id] = node
            for i, child in enumerate(node.children):
                child.index_in_parent = i
                stack.append((child, node.id))

    def get_node(self, node_id: str) -> Optional[AccessibilityNode]:
        return self.node_index.get(node_id)

    # ── Collection queries ────────────────────────────────────────────────

    def get_interactive_nodes(self) -> list[AccessibilityNode]:
        """All nodes with interactive roles."""
        return [n for n in self.node_index.values() if n.is_interactive()]

    def get_visible_nodes(self) -> list[AccessibilityNode]:
        """All visible (non-hidden) nodes."""
        return [n for n in self.node_index.values() if n.is_visible()]

    def get_focusable_nodes(self) -> list[AccessibilityNode]:
        """All nodes that can receive keyboard focus."""
        return [n for n in self.node_index.values() if n.is_focusable()]

    def get_nodes_by_role(self, role: str) -> list[AccessibilityNode]:
        return [n for n in self.node_index.values() if n.role == role]

    def get_leaves(self) -> list[AccessibilityNode]:
        """Return all leaf nodes (no children)."""
        return [n for n in self.node_index.values() if not n.children]

    # ── Metrics ───────────────────────────────────────────────────────────

    def depth(self) -> int:
        """Maximum depth across all nodes."""
        if not self.node_index:
            return 0
        return max(n.depth for n in self.node_index.values())

    def size(self) -> int:
        """Total number of nodes."""
        return len(self.node_index)

    # ── Algorithms ────────────────────────────────────────────────────────

    def lca(self, node_a_id: str, node_b_id: str) -> Optional[AccessibilityNode]:
        """Lowest common ancestor via ancestor-set intersection.

        Walks up from both nodes, intersects ancestor chains, and returns
        the deepest common ancestor.
        """
        node_a = self.get_node(node_a_id)
        node_b = self.get_node(node_b_id)
        if node_a is None or node_b is None:
            return None

        # Collect ancestors of A (including A itself)
        ancestors_a: dict[str, int] = {}  # id -> depth
        current: Optional[AccessibilityNode] = node_a
        while current is not None:
            ancestors_a[current.id] = current.depth
            pid = current.parent_id
            current = self.get_node(pid) if pid else None

        # Walk up from B and find the first hit
        best: Optional[AccessibilityNode] = None
        best_depth = -1
        current = node_b
        while current is not None:
            if current.id in ancestors_a and current.depth > best_depth:
                best = current
                best_depth = current.depth
            pid = current.parent_id
            current = self.get_node(pid) if pid else None

        return best

    def subtree(self, node_id: str) -> Optional[AccessibilityTree]:
        """Return a new AccessibilityTree rooted at the given node (deep copy)."""
        node = self.get_node(node_id)
        if node is None:
            return None
        import copy

        subtree_root = copy.deepcopy(node)
        subtree_root.parent_id = None
        return AccessibilityTree(root=subtree_root, metadata=dict(self.metadata))

    def path_between(self, src_id: str, dst_id: str) -> Optional[list[str]]:
        """Return list of node ids on the shortest path through the tree."""
        anc = self.lca(src_id, dst_id)
        if anc is None:
            return None

        def _path_to_ancestor(start_id: str, anc_id: str) -> list[str]:
            path: list[str] = []
            cid: Optional[str] = start_id
            while cid is not None and cid != anc_id:
                path.append(cid)
                node = self.get_node(cid)
                cid = node.parent_id if node else None
            path.append(anc_id)
            return path

        path_a = _path_to_ancestor(src_id, anc.id)
        path_b = _path_to_ancestor(dst_id, anc.id)
        # path_a = [src, ..., lca], path_b = [dst, ..., lca]
        return path_a + list(reversed(path_b[:-1]))

    def iter_bfs(self) -> Iterator[AccessibilityNode]:
        """Breadth-first iteration over the tree."""
        queue: deque[AccessibilityNode] = deque([self.root])
        while queue:
            node = queue.popleft()
            yield node
            queue.extend(node.children)

    def iter_dfs(self) -> Iterator[AccessibilityNode]:
        """Pre-order depth-first iteration."""
        return self.root.iter_preorder()

    # ── Validation ────────────────────────────────────────────────────────

    def validate(self) -> list[str]:
        """Basic structural validation. Returns list of error messages."""
        errors: list[str] = []
        seen_ids: set[str] = set()

        for node in self.root.iter_preorder():
            if node.id in seen_ids:
                errors.append(f"Duplicate node id: {node.id!r}")
            seen_ids.add(node.id)

            if node.parent_id is not None:
                parent = self.get_node(node.parent_id)
                if parent is None:
                    errors.append(
                        f"Node {node.id!r} references missing parent {node.parent_id!r}"
                    )
                elif not any(c.id == node.id for c in parent.children):
                    errors.append(
                        f"Node {node.id!r} not in parent {node.parent_id!r}'s children list"
                    )

            if node.bounding_box is not None:
                bb = node.bounding_box
                if bb.width < 0 or bb.height < 0:
                    errors.append(f"Node {node.id!r} has negative bounding box dimension")

        return errors

    # ── Serialisation ─────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": self.root.to_dict(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AccessibilityTree:
        root = AccessibilityNode.from_dict(data["root"])
        return cls(root=root, metadata=data.get("metadata", {}))

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> AccessibilityTree:
        return cls.from_dict(json.loads(json_str))

    def __repr__(self) -> str:
        return f"AccessibilityTree(size={self.size()}, depth={self.depth()})"
