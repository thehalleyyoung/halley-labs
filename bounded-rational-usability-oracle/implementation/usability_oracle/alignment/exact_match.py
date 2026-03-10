"""
usability_oracle.alignment.exact_match — Pass 1: exact matching.

Finds node pairs that are structurally identical across two accessibility-tree
versions.  Three sub-strategies are tried in priority order:

1. **Semantic-hash matching** — an SHA-256 digest computed from a node's role,
   name, and recursively-hashed children determines structural identity.
2. **ID matching** — nodes with the same ``node_id`` string are paired.
3. **Path matching** — nodes at the same tree-path (e.g. ``/root/nav/btn``)
   are paired.

All matches produced by this pass receive ``confidence = 1.0``.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Sequence

from usability_oracle.alignment.models import (
    AccessibilityNode,
    AccessibilityTree,
    AlignmentConfig,
    AlignmentPass,
    NodeMapping,
)


@dataclass
class _HashEntry:
    """Intermediate structure tying a node-id to its semantic hash."""

    node_id: str
    semantic_hash: str
    role: str
    name: str
    depth: int


class ExactMatcher:
    """Pass-1 exact matcher — identifies structurally identical node pairs.

    Parameters
    ----------
    config : AlignmentConfig
        Pipeline-wide configuration (controls which sub-strategies are active).
    """

    def __init__(self, config: Optional[AlignmentConfig] = None) -> None:
        self.config = config or AlignmentConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def match(
        self,
        source_tree: AccessibilityTree,
        target_tree: AccessibilityTree,
    ) -> list[NodeMapping]:
        """Run all enabled exact-matching sub-strategies.

        Returns a de-duplicated list of :class:`NodeMapping` objects.  A node
        that is matched by an earlier strategy is excluded from later ones.
        """
        matched_source: set[str] = set()
        matched_target: set[str] = set()
        all_mappings: list[NodeMapping] = []

        # --- 1. Semantic-hash matching ------------------------------------
        if self.config.enable_hash_match:
            source_hashes = self._compute_semantic_hashes(source_tree)
            target_hashes = self._compute_semantic_hashes(target_tree)
            hash_mappings = self._match_by_hash(source_hashes, target_hashes)
            for m in hash_mappings:
                if m.source_id not in matched_source and m.target_id not in matched_target:
                    matched_source.add(m.source_id)
                    matched_target.add(m.target_id)
                    all_mappings.append(m)

        # --- 2. ID matching -----------------------------------------------
        if self.config.enable_id_match:
            id_mappings = self._match_by_id(source_tree, target_tree)
            for m in id_mappings:
                if m.source_id not in matched_source and m.target_id not in matched_target:
                    matched_source.add(m.source_id)
                    matched_target.add(m.target_id)
                    all_mappings.append(m)

        # --- 3. Path matching ---------------------------------------------
        if self.config.enable_path_match:
            source_tree.compute_paths()
            target_tree.compute_paths()
            path_mappings = self._match_by_path(source_tree, target_tree)
            for m in path_mappings:
                if m.source_id not in matched_source and m.target_id not in matched_target:
                    matched_source.add(m.source_id)
                    matched_target.add(m.target_id)
                    all_mappings.append(m)

        return all_mappings

    # ------------------------------------------------------------------
    # Semantic hashing
    # ------------------------------------------------------------------

    def _compute_semantic_hashes(
        self, tree: AccessibilityTree
    ) -> dict[str, str]:
        """Compute a bottom-up semantic hash for every node in *tree*.

        The hash encodes the triple ``(role, name, [child_hashes…])`` using
        SHA-256 so that two sub-trees with identical structure produce the
        same digest.

        Returns a mapping from *node_id* → *hex-digest*.
        """
        hashes: dict[str, str] = {}
        order = tree._postorder()

        for nid in order:
            node = tree.get_node(nid)
            child_hashes = tuple(
                hashes.get(cid, "") for cid in node.children_ids if cid in tree.nodes
            )
            payload = self._hash_payload(node.role.value, node.name, child_hashes)
            digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
            hashes[nid] = digest

        return hashes

    @staticmethod
    def _hash_payload(role: str, name: str, child_hashes: tuple[str, ...]) -> str:
        """Build a deterministic string for hashing.

        Format::

            ROLE:role|NAME:name|CHILDREN:h1,h2,…
        """
        children_str = ",".join(child_hashes)
        return f"ROLE:{role}|NAME:{name}|CHILDREN:{children_str}"

    def _match_by_hash(
        self,
        source_hashes: dict[str, str],
        target_hashes: dict[str, str],
    ) -> list[NodeMapping]:
        """Pair nodes whose semantic hashes are equal.

        When multiple source nodes share the same hash (e.g. identical list
        items) they are paired by *insertion order* (which reflects document
        order from the post-order traversal).
        """
        # Invert target hash → list of node-ids
        target_by_hash: dict[str, list[str]] = defaultdict(list)
        for nid, h in target_hashes.items():
            target_by_hash[h].append(nid)

        mappings: list[NodeMapping] = []
        used_targets: set[str] = set()

        for src_nid, src_hash in source_hashes.items():
            candidates = target_by_hash.get(src_hash, [])
            for tgt_nid in candidates:
                if tgt_nid not in used_targets:
                    mappings.append(
                        NodeMapping(
                            source_id=src_nid,
                            target_id=tgt_nid,
                            confidence=1.0,
                            pass_matched=AlignmentPass.EXACT_HASH,
                        )
                    )
                    used_targets.add(tgt_nid)
                    break

        return mappings

    # ------------------------------------------------------------------
    # ID matching
    # ------------------------------------------------------------------

    def _match_by_id(
        self,
        source_tree: AccessibilityTree,
        target_tree: AccessibilityTree,
    ) -> list[NodeMapping]:
        """Pair nodes that share the same ``node_id`` across both trees.

        Only nodes whose *role* also matches are accepted — a recycled DOM id
        pointing at a completely different widget is not a genuine match.
        """
        mappings: list[NodeMapping] = []
        target_ids = set(target_tree.all_node_ids())

        for src_nid in source_tree.all_node_ids():
            if src_nid in target_ids:
                src_node = source_tree.get_node(src_nid)
                tgt_node = target_tree.get_node(src_nid)
                if src_node.role == tgt_node.role:
                    mappings.append(
                        NodeMapping(
                            source_id=src_nid,
                            target_id=src_nid,
                            confidence=1.0,
                            pass_matched=AlignmentPass.EXACT_ID,
                        )
                    )

        return mappings

    # ------------------------------------------------------------------
    # Path matching
    # ------------------------------------------------------------------

    def _match_by_path(
        self,
        source_tree: AccessibilityTree,
        target_tree: AccessibilityTree,
    ) -> list[NodeMapping]:
        """Pair nodes occupying the same tree-path in both trees.

        Requires that :meth:`AccessibilityTree.compute_paths` has been called
        so that ``tree_path`` is populated.  Matches are only accepted when the
        role also matches.
        """
        target_by_path: dict[str, str] = {}
        for nid in target_tree.all_node_ids():
            node = target_tree.get_node(nid)
            if node.tree_path:
                target_by_path[node.tree_path] = nid

        mappings: list[NodeMapping] = []
        for src_nid in source_tree.all_node_ids():
            src_node = source_tree.get_node(src_nid)
            if not src_node.tree_path:
                continue
            tgt_nid = target_by_path.get(src_node.tree_path)
            if tgt_nid is None:
                continue
            tgt_node = target_tree.get_node(tgt_nid)
            if src_node.role == tgt_node.role:
                mappings.append(
                    NodeMapping(
                        source_id=src_nid,
                        target_id=tgt_nid,
                        confidence=1.0,
                        pass_matched=AlignmentPass.EXACT_PATH,
                    )
                )

        return mappings
