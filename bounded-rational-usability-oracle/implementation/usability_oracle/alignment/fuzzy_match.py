"""
usability_oracle.alignment.fuzzy_match — Pass 2: fuzzy bipartite matching.

For nodes that were *not* paired during exact matching this pass computes a
multi-dimensional similarity matrix and solves the optimal assignment problem
with the Hungarian algorithm (``scipy.optimize.linear_sum_assignment``).

Similarity dimensions:

* **Role similarity** — taxonomy distance between WAI-ARIA roles.
* **Name similarity** — normalised Levenshtein edit distance on accessible names.
* **Position similarity** — Gaussian kernel on Euclidean bounding-box centre distance.
* **Structure similarity** — Jaccard index on multisets of child roles.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment  # type: ignore[import-untyped]

from usability_oracle.alignment.models import (
    AccessibilityNode,
    AccessibilityRole,
    AccessibilityTree,
    AlignmentConfig,
    AlignmentPass,
    BoundingBox,
    NodeMapping,
    role_taxonomy_distance,
)


# ---------------------------------------------------------------------------
# Levenshtein distance (pure-Python, no external dependency)
# ---------------------------------------------------------------------------

def _levenshtein(s: str, t: str) -> int:
    """Classic dynamic-programming Levenshtein distance."""
    n, m = len(s), len(t)
    if n == 0:
        return m
    if m == 0:
        return n

    # Use two-row optimisation for O(min(n, m)) space.
    if n < m:
        s, t = t, s
        n, m = m, n

    prev = list(range(m + 1))
    curr = [0] * (m + 1)

    for i in range(1, n + 1):
        curr[0] = i
        for j in range(1, m + 1):
            cost = 0 if s[i - 1] == t[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,       # deletion
                curr[j - 1] + 1,   # insertion
                prev[j - 1] + cost  # substitution
            )
        prev, curr = curr, prev

    return prev[m]


def _normalised_levenshtein(a: str, b: str) -> float:
    """Return a similarity in [0, 1] based on Levenshtein distance."""
    if a == b:
        return 1.0
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 1.0
    return 1.0 - _levenshtein(a, b) / max_len


# ---------------------------------------------------------------------------
# FuzzyMatcher
# ---------------------------------------------------------------------------

class FuzzyMatcher:
    """Pass-2 fuzzy matcher using bipartite optimal assignment.

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

    def match(
        self,
        source_nodes: list[AccessibilityNode],
        target_nodes: list[AccessibilityNode],
        already_matched_source: set[str] | None = None,
        already_matched_target: set[str] | None = None,
    ) -> list[NodeMapping]:
        """Compute fuzzy matches for the unmatched source and target nodes.

        Parameters
        ----------
        source_nodes, target_nodes:
            Full node lists from both trees.
        already_matched_source, already_matched_target:
            IDs already consumed by pass-1.  These are excluded from the
            similarity computation.

        Returns
        -------
        list[NodeMapping]
            New mappings discovered by the fuzzy pass.
        """
        already_matched_source = already_matched_source or set()
        already_matched_target = already_matched_target or set()

        # Filter to unmatched only
        src = [n for n in source_nodes if n.node_id not in already_matched_source]
        tgt = [n for n in target_nodes if n.node_id not in already_matched_target]

        if not src or not tgt:
            return []

        sim_matrix = self._compute_similarity_matrix(src, tgt)
        assignments = self._solve_assignment(sim_matrix)

        mappings: list[NodeMapping] = []
        for si, ti in assignments:
            score = sim_matrix[si, ti]
            if score >= self.config.fuzzy_threshold:
                mappings.append(
                    NodeMapping(
                        source_id=src[si].node_id,
                        target_id=tgt[ti].node_id,
                        confidence=float(score),
                        pass_matched=AlignmentPass.FUZZY,
                    )
                )

        return mappings

    # ------------------------------------------------------------------
    # Similarity matrix
    # ------------------------------------------------------------------

    def _compute_similarity_matrix(
        self,
        source: list[AccessibilityNode],
        target: list[AccessibilityNode],
    ) -> np.ndarray:
        """Build an |S|×|T| similarity matrix combining four dimensions.

        Each cell ``M[i, j]`` is a weighted combination:

        .. math::

            w_r · \\text{role}(s_i, t_j) + w_n · \\text{name}(s_i, t_j)
            + w_p · \\text{pos}(s_i, t_j) + w_s · \\text{struct}(s_i, t_j)
        """
        n_src = len(source)
        n_tgt = len(target)
        matrix = np.zeros((n_src, n_tgt), dtype=np.float64)

        w_r = self.config.role_weight
        w_n = self.config.name_weight
        w_p = self.config.position_weight
        w_s = self.config.structure_weight

        # Pre-compute child-role multisets for structure similarity
        src_child_roles = [self._child_role_multiset(n) for n in source]
        tgt_child_roles = [self._child_role_multiset(n) for n in target]

        for i, s_node in enumerate(source):
            for j, t_node in enumerate(target):
                role_sim = self._role_similarity(s_node.role, t_node.role)
                name_sim = self._name_similarity(s_node.name, t_node.name)
                pos_sim = self._position_similarity(s_node.bounding_box, t_node.bounding_box)
                struct_sim = self._structure_similarity_from_multisets(
                    src_child_roles[i], tgt_child_roles[j]
                )
                matrix[i, j] = w_r * role_sim + w_n * name_sim + w_p * pos_sim + w_s * struct_sim

        return matrix

    # ------------------------------------------------------------------
    # Hungarian assignment
    # ------------------------------------------------------------------

    @staticmethod
    def _solve_assignment(similarity_matrix: np.ndarray) -> list[tuple[int, int]]:
        """Solve the optimal bipartite assignment via the Hungarian algorithm.

        ``scipy.optimize.linear_sum_assignment`` minimises cost, so we negate
        the similarity matrix.

        Returns a list of ``(source_idx, target_idx)`` pairs.
        """
        cost_matrix = -similarity_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        return list(zip(row_ind.tolist(), col_ind.tolist()))

    # ------------------------------------------------------------------
    # Individual similarity dimensions
    # ------------------------------------------------------------------

    @staticmethod
    def _role_similarity(role_a: AccessibilityRole, role_b: AccessibilityRole) -> float:
        """Similarity ∈ [0, 1] based on role taxonomy distance."""
        return 1.0 - role_taxonomy_distance(role_a, role_b)

    @staticmethod
    def _name_similarity(name_a: str, name_b: str) -> float:
        """Normalised Levenshtein similarity on accessible names.

        Case-insensitive comparison; leading / trailing whitespace stripped.
        """
        a = name_a.strip().lower()
        b = name_b.strip().lower()
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        return _normalised_levenshtein(a, b)

    def _position_similarity(
        self,
        bbox_a: Optional[BoundingBox],
        bbox_b: Optional[BoundingBox],
    ) -> float:
        """Gaussian-kernel similarity on bounding-box centre distance.

        .. math::

            \\text{sim} = \\exp\\!\\left(-\\frac{d^2}{2\\sigma^2}\\right)

        where *d* is the Euclidean distance between bounding-box centres and
        σ is ``config.position_sigma``.

        If either bounding box is ``None`` a neutral 0.5 is returned.
        """
        if bbox_a is None or bbox_b is None:
            return 0.5
        d = bbox_a.center.distance_to(bbox_b.center)
        sigma = self.config.position_sigma
        return math.exp(-(d * d) / (2.0 * sigma * sigma))

    @staticmethod
    def _structure_similarity(node_a: AccessibilityNode, node_b: AccessibilityNode) -> float:
        """Jaccard similarity on the multisets of immediate child roles.

        For leaf nodes (no children) similarity is 1.0 if both are leaves,
        0.0 if only one is.
        """
        a_children = node_a.properties.get("child_roles", [])
        b_children = node_b.properties.get("child_roles", [])
        if not a_children and not b_children:
            return 1.0
        if not a_children or not b_children:
            return 0.0
        return FuzzyMatcher._jaccard_multiset(
            Counter(a_children), Counter(b_children)
        )

    @staticmethod
    def _structure_similarity_from_multisets(
        a_roles: Counter[str], b_roles: Counter[str]
    ) -> float:
        """Jaccard on pre-computed child-role multisets."""
        if not a_roles and not b_roles:
            return 1.0
        if not a_roles or not b_roles:
            return 0.0
        return FuzzyMatcher._jaccard_multiset(a_roles, b_roles)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _child_role_multiset(node: AccessibilityNode) -> Counter[str]:
        """Build a :class:`Counter` of children_ids role values.

        Since we only have IDs (not the actual child nodes) at this level, we
        fall back to the ``child_roles`` property if populated by the tree, or
        return an empty counter.
        """
        roles = node.properties.get("child_roles", [])
        return Counter(str(r) for r in roles)

    @staticmethod
    def _jaccard_multiset(a: Counter[str], b: Counter[str]) -> float:
        """Generalised Jaccard index for multisets.

        .. math::

            J(A, B) = \\frac{\\sum \\min(A_i, B_i)}{\\sum \\max(A_i, B_i)}
        """
        all_keys = set(a.keys()) | set(b.keys())
        if not all_keys:
            return 1.0
        numerator = sum(min(a.get(k, 0), b.get(k, 0)) for k in all_keys)
        denominator = sum(max(a.get(k, 0), b.get(k, 0)) for k in all_keys)
        return numerator / denominator if denominator > 0 else 0.0

    # ------------------------------------------------------------------
    # Bulk helpers (used by the differ to pre-populate child_roles)
    # ------------------------------------------------------------------

    @staticmethod
    def populate_child_roles(tree: AccessibilityTree) -> None:
        """Walk *tree* and store ``child_roles`` in each node's properties.

        This enables :meth:`_structure_similarity` to work without back-
        references to the full tree.
        """
        for nid, node in tree.nodes.items():
            child_roles: list[str] = []
            for cid in node.children_ids:
                child_node = tree.nodes.get(cid)
                if child_node is not None:
                    child_roles.append(child_node.role.value)
            node.properties["child_roles"] = child_roles

    @staticmethod
    def compute_subtree_fingerprint(tree: AccessibilityTree, node_id: str) -> str:
        """Lightweight fingerprint of a subtree for quick structure checks.

        Concatenates role values in pre-order, separated by ``/``.
        """
        parts: list[str] = []
        stack = [node_id]
        while stack:
            nid = stack.pop()
            nd = tree.nodes.get(nid)
            if nd is None:
                continue
            parts.append(nd.role.value)
            stack.extend(reversed(nd.children_ids))
        return "/".join(parts)
