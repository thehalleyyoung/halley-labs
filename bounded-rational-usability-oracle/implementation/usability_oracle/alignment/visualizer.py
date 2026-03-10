"""
usability_oracle.alignment.visualizer — Alignment visualisation utilities.

Produces human-readable text and HTML renderings of an
:class:`AlignmentResult` together with the source and target accessibility
trees.  Useful for debugging, reporting, and interactive exploration of
semantic diffs.
"""

from __future__ import annotations

import html as html_mod
from dataclasses import dataclass
from typing import Optional

from usability_oracle.alignment.models import (
    AccessibilityNode,
    AccessibilityTree,
    AlignmentPass,
    AlignmentResult,
    EditOperation,
    EditOperationType,
    NodeMapping,
)


# ---------------------------------------------------------------------------
# ANSI colour codes (for terminal output)
# ---------------------------------------------------------------------------

class _Ansi:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    GRAY = "\033[90m"


_STATUS_COLOURS = {
    "matched": _Ansi.GREEN,
    "added": _Ansi.CYAN,
    "removed": _Ansi.RED,
    "moved": _Ansi.YELLOW,
    "renamed": _Ansi.MAGENTA,
    "retyped": _Ansi.BLUE,
    "unchanged": _Ansi.DIM,
}

_STATUS_HTML_COLOURS = {
    "matched": "#2ecc71",
    "added": "#3498db",
    "removed": "#e74c3c",
    "moved": "#f39c12",
    "renamed": "#9b59b6",
    "retyped": "#2980b9",
    "unchanged": "#95a5a6",
}


# ---------------------------------------------------------------------------
# AlignmentVisualizer
# ---------------------------------------------------------------------------

class AlignmentVisualizer:
    """Renders alignment results as coloured text or HTML."""

    # ------------------------------------------------------------------
    # Plain-text rendering
    # ------------------------------------------------------------------

    def to_text(
        self,
        result: AlignmentResult,
        source: AccessibilityTree,
        target: AccessibilityTree,
        *,
        use_colour: bool = True,
    ) -> str:
        """Return a coloured text representation of the alignment diff.

        The output shows both trees side-by-side with status annotations:
        ``matched``, ``added``, ``removed``, ``moved``, ``renamed``,
        ``retyped``.
        """
        lines: list[str] = []
        lines.append(self._text_header(result, use_colour))
        lines.append("")

        # Render source tree with annotations
        lines.append(self._section("Source Tree (before)", use_colour))
        source_status = self._node_statuses_source(result, source)
        lines.extend(self._render_tree_text(source, source_status, use_colour))
        lines.append("")

        # Render target tree with annotations
        lines.append(self._section("Target Tree (after)", use_colour))
        target_status = self._node_statuses_target(result, target)
        lines.extend(self._render_tree_text(target, target_status, use_colour))
        lines.append("")

        # Edit operations
        lines.append(self._section("Edit Operations", use_colour))
        for op in result.edit_operations:
            lines.append(self._format_operation(op, use_colour))
        if not result.edit_operations:
            lines.append("  (none)")

        lines.append("")
        lines.append(self.summary_table(result))

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # HTML rendering
    # ------------------------------------------------------------------

    def to_html(
        self,
        result: AlignmentResult,
        source: AccessibilityTree,
        target: AccessibilityTree,
    ) -> str:
        """Return an HTML visual diff of the alignment."""
        parts: list[str] = [
            "<!DOCTYPE html>",
            '<html lang="en"><head><meta charset="utf-8">',
            "<title>Alignment Diff</title>",
            "<style>",
            self._css(),
            "</style></head><body>",
            '<div class="container">',
            "<h1>Semantic Alignment Diff</h1>",
            self._html_summary(result),
            '<div class="trees">',
            '<div class="tree-panel">',
            "<h2>Source (before)</h2>",
            self._render_tree_html(source, self._node_statuses_source(result, source)),
            "</div>",
            '<div class="tree-panel">',
            "<h2>Target (after)</h2>",
            self._render_tree_html(target, self._node_statuses_target(result, target)),
            "</div>",
            "</div>",
            "<h2>Edit Operations</h2>",
            self._html_operations(result),
            "</div></body></html>",
        ]
        return "\n".join(parts)

    def _render_tree_pair(
        self,
        source: AccessibilityTree,
        target: AccessibilityTree,
        mappings: list[NodeMapping],
    ) -> str:
        """Render a compact side-by-side tree pairing (plain text)."""
        lines: list[str] = []
        max_width = 50
        mapped_src_to_tgt = {m.source_id: m.target_id for m in mappings}

        for rid in source.root_ids:
            lines.extend(
                self._pair_subtree(source, target, rid, mapped_src_to_tgt, 0, max_width)
            )
        return "\n".join(lines)

    def _pair_subtree(
        self,
        source: AccessibilityTree,
        target: AccessibilityTree,
        src_id: str,
        mapping: dict[str, str],
        depth: int,
        max_width: int,
    ) -> list[str]:
        lines: list[str] = []
        s_node = source.nodes.get(src_id)
        if s_node is None:
            return lines
        indent = "  " * depth
        src_label = f"{indent}{s_node.role.value}: {s_node.name}"
        tgt_id = mapping.get(src_id)
        if tgt_id:
            t_node = target.nodes.get(tgt_id)
            tgt_label = f"{t_node.role.value}: {t_node.name}" if t_node else "???"
        else:
            tgt_label = "(removed)"
        src_col = src_label.ljust(max_width)[:max_width]
        lines.append(f"{src_col} │ {tgt_label}")
        for cid in s_node.children_ids:
            lines.extend(self._pair_subtree(source, target, cid, mapping, depth + 1, max_width))
        return lines

    # ------------------------------------------------------------------
    # Colour helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _color_node(label: str, status: str, use_colour: bool = True) -> str:
        """Wrap *label* in ANSI colour codes based on *status*."""
        if not use_colour:
            return f"[{status}] {label}"
        colour = _STATUS_COLOURS.get(status, _Ansi.RESET)
        return f"{colour}[{status}]{_Ansi.RESET} {label}"

    def summary_table(self, result: AlignmentResult) -> str:
        """Return a compact text summary table."""
        lines: list[str] = [
            "┌─────────────────────────────────────────────┐",
            "│            Alignment Summary                │",
            "├─────────────────────────┬───────────────────┤",
            f"│ Mappings                │ {len(result.mappings):>17} │",
            f"│ Edit operations         │ {len(result.edit_operations):>17} │",
            f"│ Additions               │ {len(result.additions):>17} │",
            f"│ Removals                │ {len(result.removals):>17} │",
            f"│ Edit distance           │ {result.edit_distance:>17.4f} │",
            f"│ Similarity              │ {result.similarity_score:>17.4f} │",
            f"│ Avg confidence          │ {result.average_confidence():>17.4f} │",
            "├─────────────────────────┼───────────────────┤",
        ]
        for ap in AlignmentPass:
            count = result.pass_statistics.get(ap, 0)
            lines.append(f"│ Pass: {ap.value:<18s} │ {count:>17} │")
        lines.append("└─────────────────────────┴───────────────────┘")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers — text
    # ------------------------------------------------------------------

    @staticmethod
    def _text_header(result: AlignmentResult, use_colour: bool) -> str:
        title = "Semantic Alignment Diff"
        if use_colour:
            return f"{_Ansi.BOLD}{title}{_Ansi.RESET}"
        return title

    @staticmethod
    def _section(title: str, use_colour: bool) -> str:
        if use_colour:
            return f"{_Ansi.BOLD}── {title} ──{_Ansi.RESET}"
        return f"── {title} ──"

    def _render_tree_text(
        self,
        tree: AccessibilityTree,
        statuses: dict[str, str],
        use_colour: bool,
    ) -> list[str]:
        lines: list[str] = []
        for rid in tree.root_ids:
            self._render_node_text(tree, rid, statuses, 0, lines, use_colour)
        return lines

    def _render_node_text(
        self,
        tree: AccessibilityTree,
        node_id: str,
        statuses: dict[str, str],
        depth: int,
        lines: list[str],
        use_colour: bool,
    ) -> None:
        node = tree.nodes.get(node_id)
        if node is None:
            return
        indent = "  " * depth
        label = f"{indent}{node.role.value}: {node.name}" if node.name else f"{indent}{node.role.value}"
        status = statuses.get(node_id, "unchanged")
        lines.append(self._color_node(label, status, use_colour))
        for cid in node.children_ids:
            self._render_node_text(tree, cid, statuses, depth + 1, lines, use_colour)

    def _format_operation(self, op: EditOperation, use_colour: bool) -> str:
        src = op.source_node_id or "∅"
        tgt = op.target_node_id or "∅"
        label = f"  {op.operation_type.value}: {src} → {tgt}  (cost={op.cost:.3f})"
        detail_parts: list[str] = []
        for k, v in op.details.items():
            detail_parts.append(f"{k}={v}")
        if detail_parts:
            label += f"  [{', '.join(detail_parts)}]"
        return label

    # ------------------------------------------------------------------
    # Internal helpers — HTML
    # ------------------------------------------------------------------

    @staticmethod
    def _css() -> str:
        return """
        body { font-family: 'Segoe UI', Tahoma, sans-serif; margin: 0; padding: 20px; background: #f8f9fa; }
        .container { max-width: 1400px; margin: 0 auto; }
        h1 { color: #2c3e50; }
        h2 { color: #34495e; }
        .trees { display: flex; gap: 24px; }
        .tree-panel { flex: 1; background: #fff; border-radius: 8px; padding: 16px; box-shadow: 0 2px 8px rgba(0,0,0,.1); overflow-x: auto; }
        .node { padding: 2px 4px; margin: 1px 0; font-family: monospace; font-size: 13px; border-left: 3px solid transparent; }
        .summary-table { border-collapse: collapse; margin: 12px 0; }
        .summary-table td, .summary-table th { border: 1px solid #ddd; padding: 6px 12px; }
        .summary-table th { background: #ecf0f1; }
        .ops-table { border-collapse: collapse; width: 100%; margin: 12px 0; }
        .ops-table td, .ops-table th { border: 1px solid #ddd; padding: 4px 8px; font-size: 13px; }
        .ops-table th { background: #ecf0f1; }
        """

    def _render_tree_html(self, tree: AccessibilityTree, statuses: dict[str, str]) -> str:
        parts: list[str] = ['<div class="tree">']
        for rid in tree.root_ids:
            self._render_node_html(tree, rid, statuses, 0, parts)
        parts.append("</div>")
        return "\n".join(parts)

    def _render_node_html(
        self,
        tree: AccessibilityTree,
        node_id: str,
        statuses: dict[str, str],
        depth: int,
        parts: list[str],
    ) -> None:
        node = tree.nodes.get(node_id)
        if node is None:
            return
        status = statuses.get(node_id, "unchanged")
        colour = _STATUS_HTML_COLOURS.get(status, "#95a5a6")
        indent = depth * 16
        name_esc = html_mod.escape(node.name) if node.name else ""
        role_esc = html_mod.escape(node.role.value)
        parts.append(
            f'<div class="node" style="margin-left:{indent}px; border-left-color:{colour};">'
            f'<span style="color:{colour}; font-weight:600;">[{status}]</span> '
            f"<b>{role_esc}</b>"
            f'{": " + name_esc if name_esc else ""}'
            f"</div>"
        )
        for cid in node.children_ids:
            self._render_node_html(tree, cid, statuses, depth + 1, parts)

    def _html_summary(self, result: AlignmentResult) -> str:
        rows = [
            ("Mappings", str(len(result.mappings))),
            ("Edit operations", str(len(result.edit_operations))),
            ("Additions", str(len(result.additions))),
            ("Removals", str(len(result.removals))),
            ("Edit distance", f"{result.edit_distance:.4f}"),
            ("Similarity", f"{result.similarity_score:.4f}"),
        ]
        lines = ['<table class="summary-table"><tr><th>Metric</th><th>Value</th></tr>']
        for label, val in rows:
            lines.append(f"<tr><td>{html_mod.escape(label)}</td><td>{html_mod.escape(val)}</td></tr>")
        lines.append("</table>")
        return "\n".join(lines)

    def _html_operations(self, result: AlignmentResult) -> str:
        if not result.edit_operations:
            return "<p><em>No edit operations.</em></p>"
        lines = [
            '<table class="ops-table">',
            "<tr><th>Type</th><th>Source</th><th>Target</th><th>Cost</th><th>Details</th></tr>",
        ]
        for op in result.edit_operations:
            src = html_mod.escape(op.source_node_id or "∅")
            tgt = html_mod.escape(op.target_node_id or "∅")
            det = html_mod.escape(str(op.details)) if op.details else ""
            lines.append(
                f"<tr><td>{html_mod.escape(op.operation_type.value)}</td>"
                f"<td>{src}</td><td>{tgt}</td>"
                f"<td>{op.cost:.3f}</td><td>{det}</td></tr>"
            )
        lines.append("</table>")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Status computation
    # ------------------------------------------------------------------

    def _node_statuses_source(
        self, result: AlignmentResult, tree: AccessibilityTree
    ) -> dict[str, str]:
        """Assign a status label to each source-tree node."""
        statuses: dict[str, str] = {}
        matched = {m.source_id for m in result.mappings}
        removed = set(result.removals)
        moved = {
            op.source_node_id
            for op in result.edit_operations
            if op.operation_type == EditOperationType.MOVE and op.source_node_id
        }
        renamed = {
            op.source_node_id
            for op in result.edit_operations
            if op.operation_type == EditOperationType.RENAME and op.source_node_id
        }
        retyped = {
            op.source_node_id
            for op in result.edit_operations
            if op.operation_type == EditOperationType.RETYPE and op.source_node_id
        }

        for nid in tree.all_node_ids():
            if nid in removed:
                statuses[nid] = "removed"
            elif nid in moved:
                statuses[nid] = "moved"
            elif nid in retyped:
                statuses[nid] = "retyped"
            elif nid in renamed:
                statuses[nid] = "renamed"
            elif nid in matched:
                statuses[nid] = "matched"
            else:
                statuses[nid] = "unchanged"
        return statuses

    def _node_statuses_target(
        self, result: AlignmentResult, tree: AccessibilityTree
    ) -> dict[str, str]:
        """Assign a status label to each target-tree node."""
        statuses: dict[str, str] = {}
        matched = {m.target_id for m in result.mappings}
        added = set(result.additions)
        moved = {
            op.target_node_id
            for op in result.edit_operations
            if op.operation_type == EditOperationType.MOVE and op.target_node_id
        }
        renamed = {
            op.target_node_id
            for op in result.edit_operations
            if op.operation_type == EditOperationType.RENAME and op.target_node_id
        }
        retyped = {
            op.target_node_id
            for op in result.edit_operations
            if op.operation_type == EditOperationType.RETYPE and op.target_node_id
        }

        for nid in tree.all_node_ids():
            if nid in added:
                statuses[nid] = "added"
            elif nid in moved:
                statuses[nid] = "moved"
            elif nid in retyped:
                statuses[nid] = "retyped"
            elif nid in renamed:
                statuses[nid] = "renamed"
            elif nid in matched:
                statuses[nid] = "matched"
            else:
                statuses[nid] = "unchanged"
        return statuses
