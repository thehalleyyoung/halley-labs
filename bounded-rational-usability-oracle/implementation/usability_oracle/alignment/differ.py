"""
usability_oracle.alignment.differ — Semantic diff engine.

Orchestrates the full 3-pass alignment pipeline:

1. **Pass 1 — Exact matching** (:class:`ExactMatcher`): pair structurally
   identical nodes via semantic hash, ID, and tree-path matching.
2. **Pass 2 — Fuzzy matching** (:class:`FuzzyMatcher`): pair remaining nodes
   using a multi-dimensional similarity matrix solved by the Hungarian
   algorithm.
3. **Pass 3 — Residual classification** (:class:`ResidualClassifier`):
   classify leftover nodes as moves, renames, retypes, additions, or
   removals.

The :class:`SemanticDiffer` composes these passes and computes aggregate
edit-distance and similarity metrics.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from usability_oracle.alignment.models import (
    AccessibilityNode,
    AccessibilityTree,
    AlignmentConfig,
    AlignmentContext,
    AlignmentPass,
    AlignmentResult,
    EditOperation,
    EditOperationType,
    NodeMapping,
)
from usability_oracle.alignment.exact_match import ExactMatcher
from usability_oracle.alignment.fuzzy_match import FuzzyMatcher
from usability_oracle.alignment.classifier import ResidualClassifier
from usability_oracle.alignment.cost_model import AlignmentCostModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SemanticDiffer
# ---------------------------------------------------------------------------

class SemanticDiffer:
    """Orchestrates the 3-pass semantic tree-diff pipeline.

    Usage::

        differ = SemanticDiffer()
        result = differ.diff(source_tree, target_tree)
        print(result.summary())

    Parameters
    ----------
    config : AlignmentConfig | None
        Pipeline-wide configuration.  Uses defaults when ``None``.
    """

    def __init__(self, config: Optional[AlignmentConfig] = None) -> None:
        self.config = config or AlignmentConfig()
        self._exact_matcher = ExactMatcher(self.config)
        self._fuzzy_matcher = FuzzyMatcher(self.config)
        self._classifier = ResidualClassifier(self.config)
        self._cost_model = AlignmentCostModel(self.config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def diff(
        self,
        source_tree: AccessibilityTree,
        target_tree: AccessibilityTree,
        config: Optional[AlignmentConfig] = None,
    ) -> AlignmentResult:
        """Run the full 3-pass alignment and return an :class:`AlignmentResult`.

        Parameters
        ----------
        source_tree, target_tree : AccessibilityTree
            The two tree versions to compare.
        config : AlignmentConfig | None
            Override the differ-level config for this single invocation.
        """
        cfg = config or self.config
        ctx = AlignmentContext(source_tree=source_tree, target_tree=target_tree, config=cfg)

        # Pre-compute structural metadata
        source_tree.compute_paths()
        target_tree.compute_paths()
        source_tree.compute_subtree_sizes()
        target_tree.compute_subtree_sizes()
        FuzzyMatcher.populate_child_roles(source_tree)
        FuzzyMatcher.populate_child_roles(target_tree)

        t0 = time.monotonic()

        # --- Pass 1 -------------------------------------------------------
        logger.debug("Pass 1: exact matching")
        exact_mappings = self._pass1_exact(source_tree, target_tree)
        t1 = time.monotonic()
        logger.debug("Pass 1 complete: %d mappings in %.3fs", len(exact_mappings), t1 - t0)

        # --- Pass 2 -------------------------------------------------------
        logger.debug("Pass 2: fuzzy matching")
        fuzzy_mappings = self._pass2_fuzzy(source_tree, target_tree, exact_mappings)
        t2 = time.monotonic()
        logger.debug("Pass 2 complete: %d mappings in %.3fs", len(fuzzy_mappings), t2 - t1)

        # --- Pass 3 -------------------------------------------------------
        logger.debug("Pass 3: residual classification")
        all_mappings = exact_mappings + fuzzy_mappings
        edit_ops, additions, removals = self._pass3_classify(
            source_tree, target_tree, all_mappings, ctx
        )
        t3 = time.monotonic()
        logger.debug(
            "Pass 3 complete: %d ops, %d adds, %d removes in %.3fs",
            len(edit_ops), len(additions), len(removals), t3 - t2,
        )

        # --- Detect edits within matched pairs ----------------------------
        pair_ops = self._detect_matched_pair_edits(all_mappings, source_tree, target_tree)
        all_edit_ops = edit_ops + pair_ops

        # --- Assign costs to edit operations --------------------------------
        costed_ops = self._assign_costs(all_edit_ops, source_tree, target_tree)

        # --- Compute aggregate metrics -------------------------------------
        edit_distance = self._compute_edit_distance(all_mappings, costed_ops, source_tree, target_tree)
        similarity = self._compute_similarity(all_mappings, source_tree, target_tree)

        # --- Pass statistics -----------------------------------------------
        pass_stats = self._compute_pass_statistics(all_mappings)

        result = AlignmentResult(
            mappings=all_mappings,
            edit_operations=costed_ops,
            additions=additions,
            removals=removals,
            edit_distance=edit_distance,
            similarity_score=similarity,
            pass_statistics=pass_stats,
        )

        logger.info(
            "Alignment complete: %d mappings, edit_dist=%.4f, similarity=%.4f (%.3fs total)",
            len(all_mappings), edit_distance, similarity, time.monotonic() - t0,
        )
        return result

    # ------------------------------------------------------------------
    # Pass implementations
    # ------------------------------------------------------------------

    def _pass1_exact(
        self,
        source: AccessibilityTree,
        target: AccessibilityTree,
    ) -> list[NodeMapping]:
        """Pass 1 — exact matching by hash, id, and path."""
        return self._exact_matcher.match(source, target)

    def _pass2_fuzzy(
        self,
        source: AccessibilityTree,
        target: AccessibilityTree,
        already_matched: list[NodeMapping],
    ) -> list[NodeMapping]:
        """Pass 2 — fuzzy bipartite matching on remaining nodes."""
        matched_src = {m.source_id for m in already_matched}
        matched_tgt = {m.target_id for m in already_matched}

        src_nodes = list(source.nodes.values())
        tgt_nodes = list(target.nodes.values())

        return self._fuzzy_matcher.match(
            src_nodes, tgt_nodes,
            already_matched_source=matched_src,
            already_matched_target=matched_tgt,
        )

    def _pass3_classify(
        self,
        source: AccessibilityTree,
        target: AccessibilityTree,
        matched: list[NodeMapping],
        context: AlignmentContext,
    ) -> tuple[list[EditOperation], list[str], list[str]]:
        """Pass 3 — classify unmatched residual nodes."""
        matched_src = {m.source_id for m in matched}
        matched_tgt = {m.target_id for m in matched}

        src_unmatched = [
            source.nodes[nid]
            for nid in source.all_node_ids()
            if nid not in matched_src
        ]
        tgt_unmatched = [
            target.nodes[nid]
            for nid in target.all_node_ids()
            if nid not in matched_tgt
        ]

        return self._classifier.classify(src_unmatched, tgt_unmatched, context)

    # ------------------------------------------------------------------
    # Matched-pair edit detection
    # ------------------------------------------------------------------

    def _detect_matched_pair_edits(
        self,
        mappings: list[NodeMapping],
        source: AccessibilityTree,
        target: AccessibilityTree,
    ) -> list[EditOperation]:
        """Detect edit operations within matched node pairs.

        Even if two nodes are paired, their name, bounding box, or properties
        may differ — these generate RENAME / RESIZE / MODIFY_PROPERTY ops.
        """
        ops: list[EditOperation] = []

        for m in mappings:
            s_node = source.nodes.get(m.source_id)
            t_node = target.nodes.get(m.target_id)
            if s_node is None or t_node is None:
                continue

            # Name change
            if s_node.name != t_node.name and s_node.name and t_node.name:
                ops.append(EditOperation(
                    operation_type=EditOperationType.RENAME,
                    source_node_id=s_node.node_id,
                    target_node_id=t_node.node_id,
                    cost=0.0,
                    details={"old_name": s_node.name, "new_name": t_node.name},
                ))

            # Bounding-box change
            if s_node.bounding_box and t_node.bounding_box:
                iou = s_node.bounding_box.iou(t_node.bounding_box)
                if iou < 0.95:
                    ops.append(EditOperation(
                        operation_type=EditOperationType.RESIZE,
                        source_node_id=s_node.node_id,
                        target_node_id=t_node.node_id,
                        cost=0.0,
                        details={
                            "old_bbox": {
                                "x": s_node.bounding_box.x,
                                "y": s_node.bounding_box.y,
                                "width": s_node.bounding_box.width,
                                "height": s_node.bounding_box.height,
                            },
                            "new_bbox": {
                                "x": t_node.bounding_box.x,
                                "y": t_node.bounding_box.y,
                                "width": t_node.bounding_box.width,
                                "height": t_node.bounding_box.height,
                            },
                            "iou": iou,
                        },
                    ))

            # Reorder (same parent but different sibling position)
            if (
                s_node.parent_id
                and t_node.parent_id
                and s_node.parent_id == t_node.parent_id
            ):
                s_parent = source.nodes.get(s_node.parent_id)
                t_parent = target.nodes.get(t_node.parent_id)
                if s_parent and t_parent:
                    s_idx = (
                        s_parent.children_ids.index(s_node.node_id)
                        if s_node.node_id in s_parent.children_ids
                        else -1
                    )
                    t_idx = (
                        t_parent.children_ids.index(t_node.node_id)
                        if t_node.node_id in t_parent.children_ids
                        else -1
                    )
                    if s_idx >= 0 and t_idx >= 0 and s_idx != t_idx:
                        ops.append(EditOperation(
                            operation_type=EditOperationType.REORDER,
                            source_node_id=s_node.node_id,
                            target_node_id=t_node.node_id,
                            cost=0.0,
                            details={
                                "old_index": s_idx,
                                "new_index": t_idx,
                                "parent_id": s_node.parent_id,
                            },
                        ))

        return ops

    # ------------------------------------------------------------------
    # Cost assignment
    # ------------------------------------------------------------------

    def _assign_costs(
        self,
        operations: list[EditOperation],
        source: AccessibilityTree,
        target: AccessibilityTree,
    ) -> list[EditOperation]:
        """Return a new list with costs filled in by the cost model."""
        costed: list[EditOperation] = []
        for op in operations:
            cost = self._cost_model.edit_cost(op, source, target)
            # EditOperation is frozen, so rebuild with cost
            costed.append(EditOperation(
                operation_type=op.operation_type,
                source_node_id=op.source_node_id,
                target_node_id=op.target_node_id,
                cost=cost,
                details=op.details,
            ))
        return costed

    # ------------------------------------------------------------------
    # Aggregate metrics
    # ------------------------------------------------------------------

    def _compute_edit_distance(
        self,
        mappings: list[NodeMapping],
        operations: list[EditOperation],
        source: AccessibilityTree,
        target: AccessibilityTree,
    ) -> float:
        """Compute the aggregate structural edit distance.

        Sums the costs of all edit operations.  Normalised by the total
        number of nodes across both trees to give a per-node cost.
        """
        raw = sum(op.cost for op in operations)
        total_nodes = source.node_count() + target.node_count()
        if total_nodes == 0:
            return 0.0
        return raw / (total_nodes / 2.0)

    def _compute_similarity(
        self,
        mappings: list[NodeMapping],
        source: AccessibilityTree,
        target: AccessibilityTree,
    ) -> float:
        """Compute a [0, 1] similarity score based on matched fraction.

        .. math::

            \\text{sim} = \\frac{2 \\sum_i c_i}{|S| + |T|}

        where *c_i* is the confidence of the *i*-th mapping and |S|, |T| are
        source/target node counts.
        """
        total_nodes = source.node_count() + target.node_count()
        if total_nodes == 0:
            return 1.0
        matched_weight = sum(m.confidence for m in mappings)
        return min(1.0, 2.0 * matched_weight / total_nodes)

    # ------------------------------------------------------------------
    # Pass statistics
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_pass_statistics(mappings: list[NodeMapping]) -> dict[AlignmentPass, int]:
        """Count how many mappings each pass produced."""
        stats: dict[AlignmentPass, int] = {ap: 0 for ap in AlignmentPass}
        for m in mappings:
            stats[m.pass_matched] = stats.get(m.pass_matched, 0) + 1
        return stats
