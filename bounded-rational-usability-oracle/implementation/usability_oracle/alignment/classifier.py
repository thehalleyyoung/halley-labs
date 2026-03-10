"""
usability_oracle.alignment.classifier — Pass 3: residual classification.

After exact and fuzzy matching, some source and target nodes remain unpaired.
The residual classifier inspects these nodes and classifies them into:

* **Moves** — a node whose content (role + name) appears in both trees but
  under a different parent.
* **Renames** — a node at the same structural position whose accessible name
  changed.
* **Retypes** — a node whose role changed (e.g. ``<div>`` → ``<button>``).
* **Additions** — genuinely new nodes in the target tree.
* **Removals** — nodes deleted from the source tree.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Sequence

from usability_oracle.alignment.models import (
    AccessibilityNode,
    AccessibilityTree,
    AlignmentConfig,
    AlignmentContext,
    AlignmentPass,
    EditOperation,
    EditOperationType,
    NodeMapping,
    role_taxonomy_distance,
)
from usability_oracle.alignment.fuzzy_match import _normalised_levenshtein


# ---------------------------------------------------------------------------
# Content fingerprint helper
# ---------------------------------------------------------------------------

def _content_key(node: AccessibilityNode) -> str:
    """A light fingerprint capturing role + name (ignores position)."""
    return f"{node.role.value}::{node.name.strip().lower()}"


def _structure_key(node: AccessibilityNode) -> str:
    """Fingerprint capturing role + sorted child roles."""
    child_roles = sorted(str(r) for r in node.properties.get("child_roles", []))
    return f"{node.role.value}||{'|'.join(child_roles)}"


# ---------------------------------------------------------------------------
# ResidualClassifier
# ---------------------------------------------------------------------------

class ResidualClassifier:
    """Pass-3 residual classifier for unmatched nodes.

    Parameters
    ----------
    config : AlignmentConfig
        Pipeline-wide configuration.
    """

    def __init__(self, config: Optional[AlignmentConfig] = None) -> None:
        self.config = config or AlignmentConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(
        self,
        source_unmatched: list[AccessibilityNode],
        target_unmatched: list[AccessibilityNode],
        context: AlignmentContext,
    ) -> tuple[list[EditOperation], list[str], list[str]]:
        """Classify every residual node.

        Returns
        -------
        edit_operations : list[EditOperation]
            Detected moves, renames, and retypes.
        additions : list[str]
            Target node-ids that are genuinely new.
        removals : list[str]
            Source node-ids that were genuinely deleted.
        """
        edit_ops: list[EditOperation] = []
        consumed_source: set[str] = set()
        consumed_target: set[str] = set()

        # --- detect moves -------------------------------------------------
        move_ops = self._detect_moves(source_unmatched, target_unmatched, context)
        for op in move_ops:
            if op.source_node_id:
                consumed_source.add(op.source_node_id)
            if op.target_node_id:
                consumed_target.add(op.target_node_id)
        edit_ops.extend(move_ops)

        # Remaining after moves
        remaining_src = [n for n in source_unmatched if n.node_id not in consumed_source]
        remaining_tgt = [n for n in target_unmatched if n.node_id not in consumed_target]

        # --- detect renames -----------------------------------------------
        rename_ops = self._detect_renames(remaining_src, remaining_tgt, context)
        for op in rename_ops:
            if op.source_node_id:
                consumed_source.add(op.source_node_id)
            if op.target_node_id:
                consumed_target.add(op.target_node_id)
        edit_ops.extend(rename_ops)

        remaining_src = [n for n in remaining_src if n.node_id not in consumed_source]
        remaining_tgt = [n for n in remaining_tgt if n.node_id not in consumed_target]

        # --- detect retypes -----------------------------------------------
        retype_ops = self._detect_retypes(remaining_src, remaining_tgt, context)
        for op in retype_ops:
            if op.source_node_id:
                consumed_source.add(op.source_node_id)
            if op.target_node_id:
                consumed_target.add(op.target_node_id)
        edit_ops.extend(retype_ops)

        remaining_src = [n for n in remaining_src if n.node_id not in consumed_source]
        remaining_tgt = [n for n in remaining_tgt if n.node_id not in consumed_target]

        # --- pure additions / removals ------------------------------------
        additions = self._classify_additions(remaining_tgt)
        removals = self._classify_removals(remaining_src)

        # Generate corresponding ADD / REMOVE edit ops
        for nid in additions:
            edit_ops.append(
                EditOperation(
                    operation_type=EditOperationType.ADD,
                    source_node_id=None,
                    target_node_id=nid,
                    cost=0.0,
                    details={"classification": "addition"},
                )
            )
        for nid in removals:
            edit_ops.append(
                EditOperation(
                    operation_type=EditOperationType.REMOVE,
                    source_node_id=nid,
                    target_node_id=None,
                    cost=0.0,
                    details={"classification": "removal"},
                )
            )

        return edit_ops, additions, removals

    # ------------------------------------------------------------------
    # Move detection
    # ------------------------------------------------------------------

    def _detect_moves(
        self,
        source: list[AccessibilityNode],
        target: list[AccessibilityNode],
        context: AlignmentContext,
    ) -> list[EditOperation]:
        """Detect nodes with identical content but different parents.

        A move is identified when a source node and a target node share the
        same ``(role, name)`` but their parent IDs differ.
        """
        # Build a lookup from content-key → list of nodes
        src_by_content: dict[str, list[AccessibilityNode]] = defaultdict(list)
        for n in source:
            src_by_content[_content_key(n)].append(n)

        tgt_by_content: dict[str, list[AccessibilityNode]] = defaultdict(list)
        for n in target:
            tgt_by_content[_content_key(n)].append(n)

        ops: list[EditOperation] = []
        used_src: set[str] = set()
        used_tgt: set[str] = set()

        for key, src_nodes in src_by_content.items():
            tgt_nodes = tgt_by_content.get(key, [])
            for s_node in src_nodes:
                if s_node.node_id in used_src:
                    continue
                for t_node in tgt_nodes:
                    if t_node.node_id in used_tgt:
                        continue
                    # Require that parents differ (otherwise it's not a move)
                    if s_node.parent_id != t_node.parent_id:
                        ops.append(
                            EditOperation(
                                operation_type=EditOperationType.MOVE,
                                source_node_id=s_node.node_id,
                                target_node_id=t_node.node_id,
                                cost=0.0,
                                details={
                                    "old_parent": s_node.parent_id,
                                    "new_parent": t_node.parent_id,
                                    "content_key": key,
                                },
                            )
                        )
                        used_src.add(s_node.node_id)
                        used_tgt.add(t_node.node_id)
                        break

        return ops

    # ------------------------------------------------------------------
    # Rename detection
    # ------------------------------------------------------------------

    def _detect_renames(
        self,
        source: list[AccessibilityNode],
        target: list[AccessibilityNode],
        context: AlignmentContext,
    ) -> list[EditOperation]:
        """Detect nodes at the same structural position whose name changed.

        Criteria: same role, same parent-id, and name similarity above
        ``config.rename_threshold``.
        """
        tgt_by_role_parent: dict[tuple[str, Optional[str]], list[AccessibilityNode]] = defaultdict(
            list
        )
        for n in target:
            tgt_by_role_parent[(n.role.value, n.parent_id)].append(n)

        ops: list[EditOperation] = []
        used_tgt: set[str] = set()

        for s_node in source:
            key = (s_node.role.value, s_node.parent_id)
            candidates = tgt_by_role_parent.get(key, [])
            best_sim = 0.0
            best_t: Optional[AccessibilityNode] = None
            for t_node in candidates:
                if t_node.node_id in used_tgt:
                    continue
                sim = _normalised_levenshtein(
                    s_node.name.strip().lower(),
                    t_node.name.strip().lower(),
                )
                if sim > best_sim:
                    best_sim = sim
                    best_t = t_node

            if best_t is not None and best_sim >= self.config.rename_threshold:
                if s_node.name != best_t.name:
                    ops.append(
                        EditOperation(
                            operation_type=EditOperationType.RENAME,
                            source_node_id=s_node.node_id,
                            target_node_id=best_t.node_id,
                            cost=0.0,
                            details={
                                "old_name": s_node.name,
                                "new_name": best_t.name,
                                "name_similarity": best_sim,
                            },
                        )
                    )
                    used_tgt.add(best_t.node_id)

        return ops

    # ------------------------------------------------------------------
    # Retype detection
    # ------------------------------------------------------------------

    def _detect_retypes(
        self,
        source: list[AccessibilityNode],
        target: list[AccessibilityNode],
        context: AlignmentContext,
    ) -> list[EditOperation]:
        """Detect nodes whose role changed but name and position are similar.

        A retype occurs when the name similarity is above
        ``config.retype_threshold`` and the roles differ.
        """
        ops: list[EditOperation] = []
        used_tgt: set[str] = set()

        for s_node in source:
            best_score = 0.0
            best_t: Optional[AccessibilityNode] = None
            for t_node in target:
                if t_node.node_id in used_tgt:
                    continue
                if s_node.role == t_node.role:
                    continue  # same role → not a retype
                name_sim = _normalised_levenshtein(
                    s_node.name.strip().lower(),
                    t_node.name.strip().lower(),
                )
                pos_sim = self._position_proximity(s_node, t_node)
                combined = 0.6 * name_sim + 0.4 * pos_sim
                if combined > best_score:
                    best_score = combined
                    best_t = t_node

            if best_t is not None and best_score >= self.config.retype_threshold:
                ops.append(
                    EditOperation(
                        operation_type=EditOperationType.RETYPE,
                        source_node_id=s_node.node_id,
                        target_node_id=best_t.node_id,
                        cost=0.0,
                        details={
                            "old_role": s_node.role.value,
                            "new_role": best_t.role.value,
                            "score": best_score,
                            "taxonomy_distance": role_taxonomy_distance(s_node.role, best_t.role),
                        },
                    )
                )
                used_tgt.add(best_t.node_id)

        return ops

    # ------------------------------------------------------------------
    # Additions / removals
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_additions(target_unmatched: list[AccessibilityNode]) -> list[str]:
        """Return node-ids of genuinely new target nodes."""
        return [n.node_id for n in target_unmatched]

    @staticmethod
    def _classify_removals(source_unmatched: list[AccessibilityNode]) -> list[str]:
        """Return node-ids of genuinely deleted source nodes."""
        return [n.node_id for n in source_unmatched]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _position_proximity(a: AccessibilityNode, b: AccessibilityNode) -> float:
        """Gaussian proximity on bounding-box centres (σ = 80 px)."""
        if a.bounding_box is None or b.bounding_box is None:
            return 0.5
        import math

        d = a.bounding_box.center.distance_to(b.bounding_box.center)
        sigma = 80.0
        return math.exp(-(d * d) / (2.0 * sigma * sigma))
