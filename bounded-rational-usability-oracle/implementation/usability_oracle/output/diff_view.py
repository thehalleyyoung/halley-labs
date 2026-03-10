"""Visual diff output for usability-regression comparison.

Provides :class:`DiffFormatter` for producing human-readable diffs
between two accessibility-tree analyses — side-by-side tree diffs,
cost comparison tables, and bottleneck change summaries in HTML,
ANSI (terminal), and plain-text formats.
"""

from __future__ import annotations

import html as html_mod
from dataclasses import dataclass, field
from enum import Enum, unique
from typing import Any, Dict, List, Optional, Sequence, Tuple

from usability_oracle.output.models import (
    BottleneckDescription,
    CostComparison,
    PipelineResult,
)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

@unique
class DiffStatus(Enum):
    """Status of a diff entry."""
    ADDED = "added"
    REMOVED = "removed"
    CHANGED = "changed"
    UNCHANGED = "unchanged"


@dataclass
class DiffEntry:
    """A single entry in a diff view.

    Attributes
    ----------
    label : str
        Human-readable label (e.g. node ID, bottleneck type).
    value_a : str
        Value in version A (empty string if added in B).
    value_b : str
        Value in version B (empty string if removed from A).
    status : DiffStatus
        Change status.
    metadata : dict
        Additional metadata.
    """
    label: str
    value_a: str = ""
    value_b: str = ""
    status: DiffStatus = DiffStatus.UNCHANGED
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TreeDiffResult:
    """Result of a tree diff operation.

    Attributes
    ----------
    entries : list[DiffEntry]
        Flat list of diff entries.
    summary : str
        One-line summary.
    added_count : int
    removed_count : int
    changed_count : int
    """
    entries: list[DiffEntry] = field(default_factory=list)
    summary: str = ""
    added_count: int = 0
    removed_count: int = 0
    changed_count: int = 0


@dataclass
class CostDiffResult:
    """Result of a cost diff operation."""
    entries: list[DiffEntry] = field(default_factory=list)
    total_delta: float = 0.0
    percentage_change: float = 0.0


@dataclass
class BottleneckDiffResult:
    """Result of a bottleneck diff operation."""
    new_bottlenecks: list[BottleneckDescription] = field(default_factory=list)
    resolved_bottlenecks: list[BottleneckDescription] = field(default_factory=list)
    changed_bottlenecks: list[Tuple[BottleneckDescription, BottleneckDescription]] = field(default_factory=list)
    unchanged_bottlenecks: list[BottleneckDescription] = field(default_factory=list)


# ---------------------------------------------------------------------------
# ANSI colour codes
# ---------------------------------------------------------------------------

_ANSI_RESET = "\033[0m"
_ANSI_RED = "\033[31m"
_ANSI_GREEN = "\033[32m"
_ANSI_YELLOW = "\033[33m"
_ANSI_CYAN = "\033[36m"
_ANSI_BOLD = "\033[1m"
_ANSI_DIM = "\033[2m"

_STATUS_ANSI = {
    DiffStatus.ADDED: _ANSI_GREEN,
    DiffStatus.REMOVED: _ANSI_RED,
    DiffStatus.CHANGED: _ANSI_YELLOW,
    DiffStatus.UNCHANGED: _ANSI_DIM,
}

_STATUS_SYMBOL = {
    DiffStatus.ADDED: "+",
    DiffStatus.REMOVED: "-",
    DiffStatus.CHANGED: "~",
    DiffStatus.UNCHANGED: " ",
}


# ═══════════════════════════════════════════════════════════════════════════
# DiffFormatter
# ═══════════════════════════════════════════════════════════════════════════

class DiffFormatter:
    """Visual diff output for usability-regression comparison.

    Usage::

        fmt = DiffFormatter()
        tree_diff = fmt.format_tree_diff(tree_a, tree_b)
        html = fmt.format_as_html(tree_diff)
    """

    def __init__(self, context_lines: int = 3) -> None:
        self._context_lines = context_lines

    # ------------------------------------------------------------------
    # Tree diff
    # ------------------------------------------------------------------

    def format_tree_diff(
        self,
        nodes_a: Dict[str, Dict[str, Any]],
        nodes_b: Dict[str, Dict[str, Any]],
        alignment: Optional[Dict[str, str]] = None,
    ) -> TreeDiffResult:
        """Produce a side-by-side tree diff.

        Parameters
        ----------
        nodes_a : dict
            Flat mapping id → {role, name, ...} for version A.
        nodes_b : dict
            Same for version B.
        alignment : dict, optional
            Mapping from node IDs in A to matched IDs in B.
            If None, identity matching on ID is used.

        Returns
        -------
        TreeDiffResult
        """
        if alignment is None:
            alignment = {k: k for k in nodes_a if k in nodes_b}

        entries: list[DiffEntry] = []
        added = 0
        removed = 0
        changed = 0

        matched_b: set[str] = set()

        for aid, bid in sorted(alignment.items()):
            matched_b.add(bid)
            na = nodes_a.get(aid, {})
            nb = nodes_b.get(bid, {})
            val_a = self._node_summary(na)
            val_b = self._node_summary(nb)
            if val_a == val_b:
                entries.append(DiffEntry(
                    label=aid, value_a=val_a, value_b=val_b,
                    status=DiffStatus.UNCHANGED,
                ))
            else:
                entries.append(DiffEntry(
                    label=aid, value_a=val_a, value_b=val_b,
                    status=DiffStatus.CHANGED,
                ))
                changed += 1

        # Removed (in A but not aligned)
        for aid in sorted(nodes_a):
            if aid not in alignment:
                entries.append(DiffEntry(
                    label=aid,
                    value_a=self._node_summary(nodes_a[aid]),
                    value_b="",
                    status=DiffStatus.REMOVED,
                ))
                removed += 1

        # Added (in B but not matched)
        for bid in sorted(nodes_b):
            if bid not in matched_b:
                entries.append(DiffEntry(
                    label=bid,
                    value_a="",
                    value_b=self._node_summary(nodes_b[bid]),
                    status=DiffStatus.ADDED,
                ))
                added += 1

        summary = (
            f"{added} added, {removed} removed, {changed} changed"
        )
        return TreeDiffResult(
            entries=entries, summary=summary,
            added_count=added, removed_count=removed, changed_count=changed,
        )

    # ------------------------------------------------------------------
    # Cost diff
    # ------------------------------------------------------------------

    def format_cost_diff(
        self,
        costs_a: Dict[str, float],
        costs_b: Dict[str, float],
    ) -> CostDiffResult:
        """Produce a cost comparison table.

        Parameters
        ----------
        costs_a, costs_b : dict
            Channel name → cost value.

        Returns
        -------
        CostDiffResult
        """
        all_channels = sorted(set(costs_a) | set(costs_b))
        entries: list[DiffEntry] = []
        total_delta = 0.0

        for ch in all_channels:
            va = costs_a.get(ch, 0.0)
            vb = costs_b.get(ch, 0.0)
            delta = vb - va
            total_delta += delta

            if delta > 0:
                status = DiffStatus.ADDED
            elif delta < 0:
                status = DiffStatus.REMOVED
            else:
                status = DiffStatus.UNCHANGED

            entries.append(DiffEntry(
                label=ch,
                value_a=f"{va:.4f}",
                value_b=f"{vb:.4f}",
                status=status,
                metadata={"delta": delta},
            ))

        total_a = sum(costs_a.values())
        pct = (total_delta / total_a * 100.0) if total_a != 0.0 else 0.0

        return CostDiffResult(
            entries=entries, total_delta=total_delta,
            percentage_change=pct,
        )

    # ------------------------------------------------------------------
    # Bottleneck diff
    # ------------------------------------------------------------------

    def format_bottleneck_diff(
        self,
        bottlenecks_a: Sequence[BottleneckDescription],
        bottlenecks_b: Sequence[BottleneckDescription],
    ) -> BottleneckDiffResult:
        """Produce a bottleneck change summary.

        Parameters
        ----------
        bottlenecks_a : sequence
            Bottlenecks in version A.
        bottlenecks_b : sequence
            Bottlenecks in version B.

        Returns
        -------
        BottleneckDiffResult
        """
        # Index by (type, frozenset of affected elements)
        def _key(b: BottleneckDescription) -> Tuple[str, frozenset[str]]:
            return (b.bottleneck_type.value, frozenset(b.affected_elements))

        idx_a = {_key(b): b for b in bottlenecks_a}
        idx_b = {_key(b): b for b in bottlenecks_b}

        new: list[BottleneckDescription] = []
        resolved: list[BottleneckDescription] = []
        changed: list[Tuple[BottleneckDescription, BottleneckDescription]] = []
        unchanged: list[BottleneckDescription] = []

        for k, ba in idx_a.items():
            if k in idx_b:
                bb = idx_b[k]
                if ba.severity != bb.severity or ba.cost_impact != bb.cost_impact:
                    changed.append((ba, bb))
                else:
                    unchanged.append(ba)
            else:
                resolved.append(ba)

        for k, bb in idx_b.items():
            if k not in idx_a:
                new.append(bb)

        return BottleneckDiffResult(
            new_bottlenecks=new,
            resolved_bottlenecks=resolved,
            changed_bottlenecks=changed,
            unchanged_bottlenecks=unchanged,
        )

    # ------------------------------------------------------------------
    # Highlight regressions
    # ------------------------------------------------------------------

    def highlight_regressions(
        self,
        cost_diff: CostDiffResult,
        threshold: float = 0.05,
    ) -> list[DiffEntry]:
        """Return cost-diff entries exceeding the regression threshold.

        Parameters
        ----------
        cost_diff : CostDiffResult
        threshold : float
            Minimum delta fraction to flag (default 5%).

        Returns
        -------
        list[DiffEntry]
        """
        flagged: list[DiffEntry] = []
        for entry in cost_diff.entries:
            delta = entry.metadata.get("delta", 0.0)
            va = float(entry.value_a) if entry.value_a else 0.0
            if va != 0.0 and abs(delta / va) > threshold:
                flagged.append(entry)
        return flagged

    # ------------------------------------------------------------------
    # HTML rendering
    # ------------------------------------------------------------------

    def format_as_html(
        self,
        diff: TreeDiffResult | CostDiffResult | BottleneckDiffResult,
    ) -> str:
        """Render a diff result as self-contained HTML.

        Parameters
        ----------
        diff : TreeDiffResult | CostDiffResult | BottleneckDiffResult

        Returns
        -------
        str
            HTML string.
        """
        if isinstance(diff, TreeDiffResult):
            return self._tree_diff_html(diff)
        elif isinstance(diff, CostDiffResult):
            return self._cost_diff_html(diff)
        elif isinstance(diff, BottleneckDiffResult):
            return self._bottleneck_diff_html(diff)
        return ""

    def _tree_diff_html(self, diff: TreeDiffResult) -> str:
        css = self._diff_css()
        rows = []
        for e in diff.entries:
            cls = e.status.value
            sym = _STATUS_SYMBOL[e.status]
            rows.append(
                f'<tr class="{cls}">'
                f"<td>{sym}</td>"
                f"<td>{html_mod.escape(e.label)}</td>"
                f"<td>{html_mod.escape(e.value_a)}</td>"
                f"<td>{html_mod.escape(e.value_b)}</td>"
                f"</tr>"
            )
        body = "\n".join(rows)
        return (
            f"<html><head><style>{css}</style></head><body>"
            f"<h2>Tree Diff</h2>"
            f"<p>{html_mod.escape(diff.summary)}</p>"
            f"<table><thead><tr>"
            f"<th></th><th>Node</th><th>Version A</th><th>Version B</th>"
            f"</tr></thead><tbody>{body}</tbody></table>"
            f"</body></html>"
        )

    def _cost_diff_html(self, diff: CostDiffResult) -> str:
        css = self._diff_css()
        rows = []
        for e in diff.entries:
            cls = e.status.value
            delta = e.metadata.get("delta", 0.0)
            rows.append(
                f'<tr class="{cls}">'
                f"<td>{html_mod.escape(e.label)}</td>"
                f"<td>{html_mod.escape(e.value_a)}</td>"
                f"<td>{html_mod.escape(e.value_b)}</td>"
                f"<td>{delta:+.4f}</td>"
                f"</tr>"
            )
        body = "\n".join(rows)
        return (
            f"<html><head><style>{css}</style></head><body>"
            f"<h2>Cost Diff</h2>"
            f"<p>Total delta: {diff.total_delta:+.4f} "
            f"({diff.percentage_change:+.1f}%)</p>"
            f"<table><thead><tr>"
            f"<th>Channel</th><th>A</th><th>B</th><th>Δ</th>"
            f"</tr></thead><tbody>{body}</tbody></table>"
            f"</body></html>"
        )

    def _bottleneck_diff_html(self, diff: BottleneckDiffResult) -> str:
        css = self._diff_css()
        sections: list[str] = []

        if diff.new_bottlenecks:
            items = "".join(
                f"<li class='added'>{html_mod.escape(b.description)}</li>"
                for b in diff.new_bottlenecks
            )
            sections.append(f"<h3>New Bottlenecks</h3><ul>{items}</ul>")

        if diff.resolved_bottlenecks:
            items = "".join(
                f"<li class='removed'>{html_mod.escape(b.description)}</li>"
                for b in diff.resolved_bottlenecks
            )
            sections.append(f"<h3>Resolved Bottlenecks</h3><ul>{items}</ul>")

        if diff.changed_bottlenecks:
            items = "".join(
                f"<li class='changed'>{html_mod.escape(ba.description)} → "
                f"{html_mod.escape(bb.description)}</li>"
                for ba, bb in diff.changed_bottlenecks
            )
            sections.append(f"<h3>Changed Bottlenecks</h3><ul>{items}</ul>")

        body = "\n".join(sections)
        return (
            f"<html><head><style>{css}</style></head><body>"
            f"<h2>Bottleneck Diff</h2>{body}</body></html>"
        )

    # ------------------------------------------------------------------
    # ANSI rendering
    # ------------------------------------------------------------------

    def format_as_ansi(
        self,
        diff: TreeDiffResult | CostDiffResult | BottleneckDiffResult,
    ) -> str:
        """Render a diff result with ANSI colour codes for terminals.

        Parameters
        ----------
        diff : TreeDiffResult | CostDiffResult | BottleneckDiffResult

        Returns
        -------
        str
        """
        if isinstance(diff, TreeDiffResult):
            return self._tree_diff_ansi(diff)
        elif isinstance(diff, CostDiffResult):
            return self._cost_diff_ansi(diff)
        elif isinstance(diff, BottleneckDiffResult):
            return self._bottleneck_diff_ansi(diff)
        return ""

    def _tree_diff_ansi(self, diff: TreeDiffResult) -> str:
        lines = [f"{_ANSI_BOLD}Tree Diff{_ANSI_RESET}  {diff.summary}", ""]
        for e in diff.entries:
            color = _STATUS_ANSI[e.status]
            sym = _STATUS_SYMBOL[e.status]
            line = f"{color}{sym} {e.label:<30s}  {e.value_a:<40s}  {e.value_b}{_ANSI_RESET}"
            lines.append(line)
        return "\n".join(lines)

    def _cost_diff_ansi(self, diff: CostDiffResult) -> str:
        lines = [
            f"{_ANSI_BOLD}Cost Diff{_ANSI_RESET}  "
            f"Δ={diff.total_delta:+.4f} ({diff.percentage_change:+.1f}%)",
            "",
            f"  {'Channel':<20s} {'A':>10s} {'B':>10s} {'Δ':>10s}",
            f"  {'─' * 52}",
        ]
        for e in diff.entries:
            delta = e.metadata.get("delta", 0.0)
            color = _STATUS_ANSI[e.status]
            lines.append(
                f"  {color}{e.label:<20s} {e.value_a:>10s} "
                f"{e.value_b:>10s} {delta:>+10.4f}{_ANSI_RESET}"
            )
        return "\n".join(lines)

    def _bottleneck_diff_ansi(self, diff: BottleneckDiffResult) -> str:
        lines = [f"{_ANSI_BOLD}Bottleneck Diff{_ANSI_RESET}", ""]

        if diff.new_bottlenecks:
            lines.append(f"  {_ANSI_GREEN}New ({len(diff.new_bottlenecks)}):{_ANSI_RESET}")
            for b in diff.new_bottlenecks:
                lines.append(f"    {_ANSI_GREEN}+ {b.description}{_ANSI_RESET}")

        if diff.resolved_bottlenecks:
            lines.append(f"  {_ANSI_RED}Resolved ({len(diff.resolved_bottlenecks)}):{_ANSI_RESET}")
            for b in diff.resolved_bottlenecks:
                lines.append(f"    {_ANSI_RED}- {b.description}{_ANSI_RESET}")

        if diff.changed_bottlenecks:
            lines.append(f"  {_ANSI_YELLOW}Changed ({len(diff.changed_bottlenecks)}):{_ANSI_RESET}")
            for ba, bb in diff.changed_bottlenecks:
                lines.append(
                    f"    {_ANSI_YELLOW}~ {ba.severity.value} → "
                    f"{bb.severity.value}: {bb.description}{_ANSI_RESET}"
                )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _node_summary(node: Dict[str, Any]) -> str:
        """Produce a short summary string for a node."""
        role = node.get("role", "?")
        name = node.get("name", "")
        return f"{role}({name})" if name else str(role)

    @staticmethod
    def _diff_css() -> str:
        return (
            "body { font-family: monospace; margin: 1em; }\n"
            "table { border-collapse: collapse; width: 100%; }\n"
            "th, td { border: 1px solid #ddd; padding: 4px 8px; text-align: left; }\n"
            "tr.added { background: #e6ffe6; }\n"
            "tr.removed { background: #ffe6e6; }\n"
            "tr.changed { background: #fff3cd; }\n"
            "tr.unchanged { color: #888; }\n"
            "li.added { color: #28a745; }\n"
            "li.removed { color: #dc3545; text-decoration: line-through; }\n"
            "li.changed { color: #ffc107; }\n"
        )
