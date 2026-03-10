"""
usability_oracle.visualization.tree_viz — Accessibility tree visualization.

Renders accessibility trees as text-based tree diagrams, ASCII art,
indented outlines, and structured data for external renderers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from usability_oracle.visualization.colors import Color, ColorScheme, ACCESSIBLE_PALETTE


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TreeVizConfig:
    """Configuration for tree visualization."""
    max_depth: int = 20
    max_children: int = 50
    show_ids: bool = False
    show_bounding_box: bool = False
    show_state: bool = True
    show_properties: bool = False
    show_depth_guides: bool = True
    indent_size: int = 2
    collapse_threshold: int = 10
    highlight_roles: set[str] = field(default_factory=set)
    color_scheme: Optional[ColorScheme] = None


# ---------------------------------------------------------------------------
# Node rendering helpers
# ---------------------------------------------------------------------------

def _role_icon(role: str) -> str:
    """Map accessibility role to a text icon."""
    icons = {
        "button": "🔘",
        "link": "🔗",
        "textfield": "📝",
        "checkbox": "☑",
        "radio": "◉",
        "heading": "📌",
        "image": "🖼",
        "navigation": "🧭",
        "form": "📋",
        "dialog": "💬",
        "alert": "⚠",
        "menu": "📑",
        "menuitem": "▸",
        "list": "📃",
        "listitem": "•",
        "table": "📊",
        "region": "▢",
        "search": "🔍",
        "tab": "📂",
        "tree": "🌳",
        "treeitem": "├",
        "slider": "◻",
        "combobox": "▼",
        "toolbar": "🔧",
        "separator": "─",
        "generic": "○",
    }
    return icons.get(role.lower(), "○")


def _state_indicators(state: Any) -> str:
    """Format state as compact indicators."""
    if state is None:
        return ""
    parts = []
    if getattr(state, "disabled", False):
        parts.append("⊘")
    if getattr(state, "hidden", False):
        parts.append("👁‍🗨")
    if getattr(state, "focused", False):
        parts.append("⊛")
    if getattr(state, "checked", None) is True:
        parts.append("✓")
    if getattr(state, "selected", False):
        parts.append("★")
    if getattr(state, "expanded", None) is True:
        parts.append("▾")
    elif getattr(state, "expanded", None) is False:
        parts.append("▸")
    return " ".join(parts)


def _format_bbox(bbox: Any) -> str:
    """Format bounding box compactly."""
    if bbox is None:
        return ""
    return f"[{bbox.x:.0f},{bbox.y:.0f} {bbox.width:.0f}×{bbox.height:.0f}]"


# ---------------------------------------------------------------------------
# TreeVisualizer
# ---------------------------------------------------------------------------

class TreeVisualizer:
    """Render accessibility trees in various text formats.

    Parameters:
        config: Visualization configuration.
    """

    def __init__(self, config: TreeVizConfig | None = None) -> None:
        self._config = config or TreeVizConfig()

    # ------------------------------------------------------------------
    # ASCII tree
    # ------------------------------------------------------------------

    def render_ascii(self, tree: Any) -> str:
        """Render the tree as an ASCII art diagram.

        Example output::

            ┬ form "Login Form"
            ├─┬ region "Username"
            │ ├── textfield "Username"
            │ └── generic "Help text"
            ├─┬ region "Password"
            │ └── textfield "Password"
            └── button "Submit"
        """
        lines: list[str] = []
        self._render_node_ascii(tree.root, lines, prefix="", is_last=True, depth=0)
        return "\n".join(lines)

    def _render_node_ascii(
        self,
        node: Any,
        lines: list[str],
        prefix: str,
        is_last: bool,
        depth: int,
    ) -> None:
        if depth > self._config.max_depth:
            lines.append(f"{prefix}{'└' if is_last else '├'}── ... (truncated)")
            return

        connector = "└" if is_last else "├"
        has_children = bool(node.children) and depth < self._config.max_depth
        branch = "─┬" if has_children else "──"

        label = self._format_node_label(node)
        if depth == 0:
            lines.append(f"┬ {label}" if has_children else f"─ {label}")
        else:
            lines.append(f"{prefix}{connector}{branch} {label}")

        if not node.children:
            return

        children = node.children
        if len(children) > self._config.max_children:
            children = children[:self._config.max_children]
            truncated = True
        else:
            truncated = False

        new_prefix = prefix + ("  " if is_last else "│ ")
        for i, child in enumerate(children):
            child_is_last = (i == len(children) - 1) and not truncated
            self._render_node_ascii(child, lines, new_prefix, child_is_last, depth + 1)

        if truncated:
            omitted = len(node.children) - self._config.max_children
            lines.append(f"{new_prefix}└── ... ({omitted} more children)")

    # ------------------------------------------------------------------
    # Indented outline
    # ------------------------------------------------------------------

    def render_outline(self, tree: Any) -> str:
        """Render the tree as an indented outline."""
        lines: list[str] = []
        self._render_node_outline(tree.root, lines, depth=0)
        return "\n".join(lines)

    def _render_node_outline(self, node: Any, lines: list[str], depth: int) -> None:
        if depth > self._config.max_depth:
            return
        indent = " " * (depth * self._config.indent_size)
        label = self._format_node_label(node)
        lines.append(f"{indent}{label}")
        for child in node.children[:self._config.max_children]:
            self._render_node_outline(child, lines, depth + 1)

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------

    def render_summary(self, tree: Any) -> str:
        """Render tree summary statistics."""
        stats = self._compute_stats(tree.root)
        lines = [
            f"Accessibility Tree Summary",
            f"  Total nodes:      {stats['total']}",
            f"  Max depth:        {stats['max_depth']}",
            f"  Interactive:      {stats['interactive']}",
            f"  Landmarks:        {stats['landmarks']}",
            f"  Branching factor: {stats['avg_branching']:.1f}",
            f"  Roles:",
        ]
        for role, count in sorted(stats["role_counts"].items(), key=lambda x: -x[1]):
            lines.append(f"    {role}: {count}")
        return "\n".join(lines)

    def _compute_stats(self, root: Any) -> dict[str, Any]:
        """Compute tree statistics."""
        total = 0
        max_depth = 0
        interactive = 0
        landmarks = 0
        role_counts: dict[str, int] = {}
        child_counts: list[int] = []
        interactive_roles = {"button", "link", "textfield", "checkbox", "radio", "menuitem", "tab"}
        landmark_roles = {"navigation", "banner", "main", "contentinfo", "complementary", "search", "form", "region"}

        def _walk(node: Any, depth: int) -> None:
            nonlocal total, max_depth, interactive, landmarks
            total += 1
            max_depth = max(max_depth, depth)
            role = node.role.lower() if isinstance(node.role, str) else str(node.role).lower()
            role_counts[role] = role_counts.get(role, 0) + 1
            if role in interactive_roles:
                interactive += 1
            if role in landmark_roles:
                landmarks += 1
            child_counts.append(len(node.children))
            for child in node.children:
                _walk(child, depth + 1)

        _walk(root, 0)
        avg_branching = sum(child_counts) / max(len([c for c in child_counts if c > 0]), 1)

        return {
            "total": total,
            "max_depth": max_depth,
            "interactive": interactive,
            "landmarks": landmarks,
            "avg_branching": avg_branching,
            "role_counts": role_counts,
        }

    # ------------------------------------------------------------------
    # Diff visualization
    # ------------------------------------------------------------------

    def render_diff(self, tree_a: Any, tree_b: Any) -> str:
        """Render a side-by-side diff of two trees."""
        stats_a = self._compute_stats(tree_a.root)
        stats_b = self._compute_stats(tree_b.root)

        lines = ["Tree Comparison:", ""]
        comparisons = [
            ("Total nodes", stats_a["total"], stats_b["total"]),
            ("Max depth", stats_a["max_depth"], stats_b["max_depth"]),
            ("Interactive", stats_a["interactive"], stats_b["interactive"]),
            ("Landmarks", stats_a["landmarks"], stats_b["landmarks"]),
            ("Branching", f"{stats_a['avg_branching']:.1f}", f"{stats_b['avg_branching']:.1f}"),
        ]

        lines.append(f"  {'Metric':<20} {'Before':>10} {'After':>10} {'Change':>10}")
        lines.append(f"  {'─' * 20} {'─' * 10} {'─' * 10} {'─' * 10}")
        for name, before, after in comparisons:
            if isinstance(before, (int, float)) and isinstance(after, (int, float)):
                diff = after - before
                sign = "+" if diff > 0 else ""
                lines.append(f"  {name:<20} {before:>10} {after:>10} {sign}{diff:>9}")
            else:
                lines.append(f"  {name:<20} {before:>10} {after:>10}")

        # Role changes
        all_roles = set(stats_a["role_counts"].keys()) | set(stats_b["role_counts"].keys())
        role_changes = []
        for role in sorted(all_roles):
            ca = stats_a["role_counts"].get(role, 0)
            cb = stats_b["role_counts"].get(role, 0)
            if ca != cb:
                role_changes.append((role, ca, cb))

        if role_changes:
            lines.append("")
            lines.append("  Role Changes:")
            for role, ca, cb in role_changes:
                diff = cb - ca
                sign = "+" if diff > 0 else ""
                lines.append(f"    {role:<18} {ca:>4} → {cb:>4} ({sign}{diff})")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Node label formatting
    # ------------------------------------------------------------------

    def _format_node_label(self, node: Any) -> str:
        """Format a single node's label."""
        parts = []

        icon = _role_icon(node.role if isinstance(node.role, str) else str(node.role))
        role_str = node.role if isinstance(node.role, str) else str(node.role)
        parts.append(f"{icon} {role_str}")

        name = getattr(node, "name", "")
        if name:
            display_name = name if len(name) <= 40 else name[:37] + "..."
            parts.append(f'"{display_name}"')

        if self._config.show_ids:
            parts.append(f"[{node.id}]")

        if self._config.show_state:
            state_str = _state_indicators(getattr(node, "state", None))
            if state_str:
                parts.append(state_str)

        if self._config.show_bounding_box:
            bbox_str = _format_bbox(getattr(node, "bounding_box", None))
            if bbox_str:
                parts.append(bbox_str)

        return " ".join(parts)

    # ------------------------------------------------------------------
    # Data export for external renderers
    # ------------------------------------------------------------------

    def to_dict(self, tree: Any) -> dict[str, Any]:
        """Export tree as a nested dict for JSON serialization."""
        def _node_to_dict(node: Any) -> dict[str, Any]:
            d: dict[str, Any] = {
                "id": node.id,
                "role": node.role if isinstance(node.role, str) else str(node.role),
                "name": getattr(node, "name", ""),
            }
            bbox = getattr(node, "bounding_box", None)
            if bbox:
                d["bbox"] = {"x": bbox.x, "y": bbox.y, "w": bbox.width, "h": bbox.height}
            if node.children:
                d["children"] = [_node_to_dict(c) for c in node.children]
            return d

        return _node_to_dict(tree.root)
