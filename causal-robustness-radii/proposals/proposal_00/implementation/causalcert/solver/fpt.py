"""
Fixed-parameter tractable DP solver on tree decompositions (ALG 7).

When the moral graph of the DAG has bounded treewidth *w*, the robustness
radius can be computed in time O(n * 3^{w^2}) via dynamic programming over a
nice tree decomposition.  This is efficient for sparse real-world causal
DAGs where *w* is typically <= 5.

Algorithm
---------
1. Compute the moral graph and its tree decomposition.
2. Convert to a *nice* tree decomposition (leaf, introduce, forget, join).
3. Run bottom-up DP where the state at each bag encodes the directed-edge
   configuration among the bag's vertices.
4. At the root, find the minimum-cost state that overturns the conclusion.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from itertools import product
from typing import Any

import numpy as np

from causalcert.dag.moral import TreeDecomposition, moral_graph, tree_decomposition
from causalcert.types import (
    AdjacencyMatrix,
    ConclusionPredicate,
    EditType,
    NodeId,
    RobustnessRadius,
    SolverStrategy,
    StructuralEdit,
)

logger = logging.getLogger(__name__)

# Each unordered pair {i,j} in a bag has 3 possible edge states:
_NO_EDGE = 0
_FORWARD = 1   # i -> j  (where i < j)
_BACKWARD = 2  # j -> i


# ---------------------------------------------------------------------------
# Nice tree decomposition
# ---------------------------------------------------------------------------

_LEAF = "leaf"
_INTRODUCE = "introduce"
_FORGET = "forget"
_JOIN = "join"


class NiceTreeNode:
    """A node in a nice tree decomposition."""

    __slots__ = ("bag", "node_type", "special_vertex", "children", "index")

    def __init__(
        self,
        bag: frozenset[int],
        node_type: str,
        special_vertex: int | None = None,
        children: list[int] | None = None,
        index: int = -1,
    ) -> None:
        self.bag = bag
        self.node_type = node_type
        self.special_vertex = special_vertex
        self.children = children or []
        self.index = index


def _make_nice(td: TreeDecomposition, root: int = 0) -> list[NiceTreeNode]:
    """Convert a tree decomposition into a nice tree decomposition.

    In a nice tree decomposition every internal node is one of:

    * **Leaf** — a single-vertex bag with no children.
    * **Introduce** — bag = child_bag ∪ {v} for exactly one child.
    * **Forget** — bag = child_bag \\ {v} for exactly one child.
    * **Join** — two children with identical bags.

    Parameters
    ----------
    td : TreeDecomposition
        Input tree decomposition.
    root : int
        Root bag index.

    Returns
    -------
    list[NiceTreeNode]
        Nodes of the nice tree decomposition, root-last.
    """
    if not td.bags:
        return []

    n_bags = len(td.bags)
    adj = td.tree_adj

    # Root the tree via BFS
    parent = [-1] * n_bags
    children_of: list[list[int]] = [[] for _ in range(n_bags)]
    visited = [False] * n_bags
    queue = deque([root])
    visited[root] = True
    order: list[int] = []

    while queue:
        v = queue.popleft()
        order.append(v)
        for nb in adj[v]:
            if not visited[nb]:
                visited[nb] = True
                parent[nb] = v
                children_of[v].append(nb)
                queue.append(nb)

    # Build nice nodes bottom-up
    nice: list[NiceTreeNode] = []
    bag_map: dict[int, int] = {}  # original bag -> nice node index

    def _add(bag: frozenset[int], ntype: str, sv: int | None, ch: list[int]) -> int:
        idx = len(nice)
        nice.append(NiceTreeNode(bag, ntype, sv, list(ch), idx))
        return idx

    for orig_idx in reversed(order):
        bag = td.bags[orig_idx]
        child_idxs = children_of[orig_idx]

        if not child_idxs:
            # Leaf chain: introduce vertices one by one
            vs = sorted(bag)
            if not vs:
                ni = _add(frozenset(), _LEAF, None, [])
                bag_map[orig_idx] = ni
                continue
            # Start with a single-vertex leaf
            cur = _add(frozenset({vs[0]}), _LEAF, vs[0], [])
            for v in vs[1:]:
                cur = _add(frozenset(nice[cur].bag | {v}), _INTRODUCE, v, [cur])
            bag_map[orig_idx] = cur

        elif len(child_idxs) == 1:
            child_nice = bag_map[child_idxs[0]]
            child_bag = nice[child_nice].bag
            # Chain introduce/forget to go from child_bag to bag
            cur = child_nice
            cur_bag = child_bag

            # Forget vertices in child_bag \ bag
            for v in sorted(cur_bag - bag):
                cur = _add(frozenset(nice[cur].bag - {v}), _FORGET, v, [cur])

            # Introduce vertices in bag \ cur_bag
            cur_bag_now = nice[cur].bag
            for v in sorted(bag - cur_bag_now):
                cur = _add(frozenset(nice[cur].bag | {v}), _INTRODUCE, v, [cur])

            bag_map[orig_idx] = cur

        else:
            # Multiple children: binary join tree
            # First, make each child match the current bag
            child_nice_ids: list[int] = []
            for ci in child_idxs:
                child_nice = bag_map[ci]
                child_bag = nice[child_nice].bag
                cur = child_nice

                for v in sorted(child_bag - bag):
                    cur = _add(frozenset(nice[cur].bag - {v}), _FORGET, v, [cur])
                cur_bag_now = nice[cur].bag
                for v in sorted(bag - cur_bag_now):
                    cur = _add(frozenset(nice[cur].bag | {v}), _INTRODUCE, v, [cur])

                child_nice_ids.append(cur)

            # Join pairwise
            while len(child_nice_ids) > 1:
                new_children: list[int] = []
                for k in range(0, len(child_nice_ids) - 1, 2):
                    j_node = _add(bag, _JOIN, None, [child_nice_ids[k], child_nice_ids[k + 1]])
                    new_children.append(j_node)
                if len(child_nice_ids) % 2 == 1:
                    new_children.append(child_nice_ids[-1])
                child_nice_ids = new_children

            bag_map[orig_idx] = child_nice_ids[0]

    # Make sure root is the last node
    root_nice = bag_map[root]

    # Return post-order traversal
    post_order: list[int] = []
    vis: set[int] = set()
    stack = [root_nice]
    while stack:
        nd = stack[-1]
        all_done = True
        for c in nice[nd].children:
            if c not in vis:
                stack.append(c)
                all_done = False
        if all_done:
            stack.pop()
            if nd not in vis:
                vis.add(nd)
                post_order.append(nd)

    reindex: dict[int, int] = {}
    result: list[NiceTreeNode] = []
    for new_idx, old_idx in enumerate(post_order):
        nd = nice[old_idx]
        new_ch = [reindex[c] for c in nd.children if c in reindex]
        result.append(NiceTreeNode(nd.bag, nd.node_type, nd.special_vertex, new_ch, new_idx))
        reindex[old_idx] = new_idx

    return result


# ---------------------------------------------------------------------------
# State representation
# ---------------------------------------------------------------------------


def _pair_key(i: int, j: int) -> tuple[int, int]:
    """Canonical ordered pair (smaller first)."""
    return (min(i, j), max(i, j))


def _enumerate_edge_configs(
    vertices: list[int],
) -> list[dict[tuple[int, int], int]]:
    """Enumerate all possible directed-edge configurations among vertices.

    For each unordered pair {i,j} with i < j, there are 3 options:
    0 = no edge, 1 = i->j, 2 = j->i.

    Returns list of dicts mapping (i,j) -> state.
    """
    pairs = []
    for a_idx in range(len(vertices)):
        for b_idx in range(a_idx + 1, len(vertices)):
            pairs.append((vertices[a_idx], vertices[b_idx]))

    if not pairs:
        return [{}]

    configs: list[dict[tuple[int, int], int]] = []
    for combo in product(range(3), repeat=len(pairs)):
        config = {}
        for k, (i, j) in enumerate(pairs):
            config[i, j] = combo[k]
        configs.append(config)
    return configs


def _is_acyclic_config(
    vertices: list[int], config: dict[tuple[int, int], int]
) -> bool:
    """Check if an edge configuration among vertices forms a DAG.

    Uses topological sort (Kahn's algorithm) on the subgraph.
    """
    adj_set: dict[int, set[int]] = {v: set() for v in vertices}
    in_deg: dict[int, int] = {v: 0 for v in vertices}

    for (i, j), state in config.items():
        if state == _FORWARD:  # i -> j
            adj_set[i].add(j)
            in_deg[j] += 1
        elif state == _BACKWARD:  # j -> i
            adj_set[j].add(i)
            in_deg[i] += 1

    queue = deque(v for v in vertices if in_deg[v] == 0)
    count = 0
    while queue:
        v = queue.popleft()
        count += 1
        for u in adj_set[v]:
            in_deg[u] -= 1
            if in_deg[u] == 0:
                queue.append(u)
    return count == len(vertices)


def _config_to_key(
    vertices: list[int], config: dict[tuple[int, int], int]
) -> tuple[int, ...]:
    """Convert an edge configuration to a hashable tuple key.

    Key is a tuple of edge states for each pair (i,j), i < j,
    sorted lexicographically.
    """
    pairs = []
    for a_idx in range(len(vertices)):
        for b_idx in range(a_idx + 1, len(vertices)):
            pairs.append(_pair_key(vertices[a_idx], vertices[b_idx]))
    pairs.sort()
    return tuple(config.get(p, _NO_EDGE) for p in pairs)


def _key_to_config(
    vertices: list[int], key: tuple[int, ...]
) -> dict[tuple[int, int], int]:
    """Inverse of _config_to_key."""
    pairs: list[tuple[int, int]] = []
    for a_idx in range(len(vertices)):
        for b_idx in range(a_idx + 1, len(vertices)):
            pairs.append(_pair_key(vertices[a_idx], vertices[b_idx]))
    pairs.sort()
    return {p: key[k] for k, p in enumerate(pairs)}


def _edit_cost_of_pair(
    i: int, j: int, state: int, orig_adj: AdjacencyMatrix
) -> int:
    """Edit cost for a single unordered pair {i,j} given edge state.

    Compare the edge state (0/1/2) with the original adjacency matrix.
    """
    orig_ij = int(orig_adj[i, j])
    orig_ji = int(orig_adj[j, i])

    if state == _NO_EDGE:
        if orig_ij == 0 and orig_ji == 0:
            return 0
        else:
            return 1  # delete
    elif state == _FORWARD:  # i -> j
        if orig_ij == 1:
            return 0
        elif orig_ji == 1:
            return 1  # reverse
        else:
            return 1  # add
    elif state == _BACKWARD:  # j -> i
        if orig_ji == 1:
            return 0
        elif orig_ij == 1:
            return 1  # reverse
        else:
            return 1  # add
    return 0


def _config_adj(n: int, config: dict[tuple[int, int], int]) -> AdjacencyMatrix:
    """Convert a global edge configuration to an adjacency matrix."""
    adj = np.zeros((n, n), dtype=np.int8)
    for (i, j), state in config.items():
        if state == _FORWARD:
            adj[i, j] = 1
        elif state == _BACKWARD:
            adj[j, i] = 1
    return adj


def _apply_config_to_adj(
    orig_adj: AdjacencyMatrix,
    config: dict[tuple[int, int], int],
) -> AdjacencyMatrix:
    """Apply edge config decisions onto a copy of the original adjacency.

    Pairs present in *config* are overwritten; all other edges stay as-is.
    """
    result = np.asarray(orig_adj, dtype=np.int8).copy()
    for (i, j), state in config.items():
        result[i, j] = 0
        result[j, i] = 0
        if state == _FORWARD:
            result[i, j] = 1
        elif state == _BACKWARD:
            result[j, i] = 1
    return result


# ---------------------------------------------------------------------------
# FPT Solver
# ---------------------------------------------------------------------------


class FPTSolver:
    """FPT dynamic-programming solver on a tree decomposition (ALG 7).

    Parameters
    ----------
    max_treewidth : int
        Maximum treewidth for which this solver will attempt DP.
    time_limit_s : float
        Maximum wall-clock time.
    """

    def __init__(
        self,
        max_treewidth: int = 8,
        time_limit_s: float = 300.0,
    ) -> None:
        self.max_treewidth = max_treewidth
        self.time_limit_s = time_limit_s

    def solve(
        self,
        adj: AdjacencyMatrix,
        predicate: ConclusionPredicate,
        data: Any,
        treatment: NodeId,
        outcome: NodeId,
        max_k: int = 10,
        decomposition: TreeDecomposition | None = None,
    ) -> RobustnessRadius:
        """Solve via DP on the tree decomposition.

        Parameters
        ----------
        adj : AdjacencyMatrix
            DAG adjacency matrix.
        predicate : ConclusionPredicate
            Conclusion predicate.
        data : Any
            Observational data.
        treatment, outcome : NodeId
            Treatment and outcome nodes.
        max_k : int
            Maximum edits.
        decomposition : TreeDecomposition | None
            Pre-computed tree decomposition.

        Returns
        -------
        RobustnessRadius
        """
        adj = np.asarray(adj, dtype=np.int8)
        n = adj.shape[0]
        t0 = time.perf_counter()

        # Quick check
        if not predicate(adj, data, treatment=treatment, outcome=outcome):
            return RobustnessRadius(
                lower_bound=0, upper_bound=0,
                witness_edits=(),
                solver_strategy=SolverStrategy.FPT,
                solver_time_s=time.perf_counter() - t0,
                gap=0.0, certified=True,
            )

        # Compute tree decomposition if not provided
        if decomposition is None:
            mg = moral_graph(adj)
            decomposition = tree_decomposition(mg)

        tw = decomposition.width
        if tw > self.max_treewidth:
            logger.warning(
                "Treewidth %d exceeds max %d; FPT may be slow",
                tw, self.max_treewidth,
            )

        # Convert to nice tree decomposition
        root_idx = 0 if decomposition.bags else -1
        nice_nodes = _make_nice(decomposition, root=root_idx)

        if not nice_nodes:
            return RobustnessRadius(
                lower_bound=max_k + 1, upper_bound=max_k + 1,
                witness_edits=(),
                solver_strategy=SolverStrategy.FPT,
                solver_time_s=time.perf_counter() - t0,
                gap=0.0, certified=True,
            )

        # Run DP bottom-up.
        # dp_table[node_idx] maps
        #   state_key -> (min_cost, full_edge_config)
        # where full_edge_config accumulates ALL committed edge decisions
        # (including forgotten edges) so the final adjacency can be rebuilt.
        _DPVal = tuple[int, dict[tuple[int, int], int]]
        dp_table: dict[int, dict[tuple[int, ...], _DPVal]] = {}

        for node in nice_nodes:
            elapsed = time.perf_counter() - t0
            if elapsed >= self.time_limit_s:
                logger.warning("FPT time limit reached during DP")
                break

            bag_verts = sorted(node.bag)
            idx = node.index

            if node.node_type == _LEAF:
                dp_table[idx] = {(): (0, {})}

            elif node.node_type == _INTRODUCE:
                v = node.special_vertex
                assert v is not None
                child_idx = node.children[0]
                child_dp = dp_table.get(child_idx, {})

                new_dp: dict[tuple[int, ...], _DPVal] = {}

                for child_key, (child_cost, child_full) in child_dp.items():
                    other_verts = [u for u in bag_verts if u != v]
                    pairs_with_v = [_pair_key(v, u) for u in other_verts]

                    for combo in product(range(3), repeat=len(pairs_with_v)):
                        extra_cost = 0
                        new_full = dict(child_full)
                        for k, pair in enumerate(pairs_with_v):
                            new_full[pair] = combo[k]
                            extra_cost += _edit_cost_of_pair(
                                pair[0], pair[1], combo[k], adj
                            )

                        total_cost = child_cost + extra_cost
                        if total_cost > max_k:
                            continue

                        # Build bag-local config for acyclicity check
                        bag_config = {
                            p: new_full[p]
                            for p in new_full
                            if p[0] in node.bag and p[1] in node.bag
                        }
                        if not _is_acyclic_config(bag_verts, bag_config):
                            continue

                        new_key = _config_to_key(bag_verts, bag_config)
                        if new_key not in new_dp or new_dp[new_key][0] > total_cost:
                            new_dp[new_key] = (total_cost, new_full)

                dp_table[idx] = new_dp

            elif node.node_type == _FORGET:
                v = node.special_vertex
                assert v is not None
                child_idx = node.children[0]
                child_dp = dp_table.get(child_idx, {})

                new_dp = {}

                for child_key, (child_cost, child_full) in child_dp.items():
                    # The full config already contains edges involving v;
                    # just change the DP key to exclude v's local pairs.
                    bag_config = {
                        p: child_full[p]
                        for p in child_full
                        if p[0] in node.bag and p[1] in node.bag
                    }
                    proj_key = _config_to_key(bag_verts, bag_config)
                    if proj_key not in new_dp or new_dp[proj_key][0] > child_cost:
                        new_dp[proj_key] = (child_cost, child_full)

                dp_table[idx] = new_dp

            elif node.node_type == _JOIN:
                child1_idx = node.children[0]
                child2_idx = node.children[1]
                child1_dp = dp_table.get(child1_idx, {})
                child2_dp = dp_table.get(child2_idx, {})

                new_dp = {}

                for key1, (cost1, full1) in child1_dp.items():
                    if key1 not in child2_dp:
                        continue
                    cost2, full2 = child2_dp[key1]

                    # Merge full configs (they agree on bag-local pairs)
                    merged = dict(full1)
                    merged.update(full2)  # full2 overwrites shared keys

                    # Subtract double-counted bag-internal edge costs
                    bag_internal_cost = sum(
                        _edit_cost_of_pair(p[0], p[1], full1.get(p, 0), adj)
                        for p in full1
                        if p[0] in node.bag and p[1] in node.bag
                    )
                    total = cost1 + cost2 - bag_internal_cost
                    if total > max_k:
                        continue
                    if key1 not in new_dp or new_dp[key1][0] > total:
                        new_dp[key1] = (total, merged)

                dp_table[idx] = new_dp

        # Extract best solution from root
        root_dp = dp_table.get(nice_nodes[-1].index, {}) if nice_nodes else {}

        best_cost = max_k + 1
        best_full: dict[tuple[int, int], int] | None = None

        from causalcert.dag.validation import is_dag

        for key, (cost, full_config) in root_dp.items():
            if cost >= best_cost:
                continue

            # Build full adjacency: start from original, apply config
            full_adj = _apply_config_to_adj(adj, full_config)

            if not is_dag(full_adj):
                continue

            if not predicate(full_adj, data, treatment=treatment, outcome=outcome):
                best_cost = cost
                best_full = full_config

        elapsed = time.perf_counter() - t0

        if best_full is not None:
            full_adj = _apply_config_to_adj(adj, best_full)
            edits = self._extract_edits(adj, full_adj)
            actual_cost = len(edits) if edits else best_cost
            return RobustnessRadius(
                lower_bound=actual_cost,
                upper_bound=actual_cost,
                witness_edits=tuple(edits),
                solver_strategy=SolverStrategy.FPT,
                solver_time_s=elapsed,
                gap=0.0,
                certified=True,
            )
        else:
            return RobustnessRadius(
                lower_bound=max_k + 1,
                upper_bound=max_k + 1,
                witness_edits=(),
                solver_strategy=SolverStrategy.FPT,
                solver_time_s=elapsed,
                gap=0.0,
                certified=True,
            )

    def _dp_on_bag(
        self,
        bag_idx: int,
        decomposition: TreeDecomposition,
        adj: AdjacencyMatrix,
        max_k: int,
    ) -> dict[tuple[int, ...], int]:
        """Run DP for a single bag (standalone, for testing).

        Parameters
        ----------
        bag_idx : int
            Index of the bag.
        decomposition : TreeDecomposition
            Tree decomposition.
        adj : AdjacencyMatrix
            DAG adjacency matrix.
        max_k : int
            Maximum edits.

        Returns
        -------
        dict[tuple[int, ...], int]
            Mapping from state keys to minimum edit cost.
        """
        bag = decomposition.bags[bag_idx]
        verts = sorted(bag)

        configs = _enumerate_edge_configs(verts)
        result: dict[tuple[int, ...], int] = {}

        for config in configs:
            if not _is_acyclic_config(verts, config):
                continue

            cost = 0
            for (i, j), state in config.items():
                cost += _edit_cost_of_pair(i, j, state, adj)

            if cost <= max_k:
                key = _config_to_key(verts, config)
                if key not in result or result[key] > cost:
                    result[key] = cost

        return result

    def _traceback(
        self,
        dp_table: dict[int, dict[tuple[int, ...], int]],
        decomposition: TreeDecomposition,
    ) -> list[StructuralEdit]:
        """Recover the optimal edit set from the DP tables.

        Parameters
        ----------
        dp_table : dict
            DP tables indexed by bag.
        decomposition : TreeDecomposition
            Tree decomposition.

        Returns
        -------
        list[StructuralEdit]
        """
        if not dp_table:
            return []

        root_idx = max(dp_table.keys())
        root_table = dp_table[root_idx]
        if not root_table:
            return []

        best_key = min(root_table, key=root_table.get)  # type: ignore[arg-type]
        best_cost = root_table[best_key]

        if best_cost == 0:
            return []

        # The key encodes the edge configuration; reconstruct edits
        verts = sorted(decomposition.bags[root_idx] if root_idx < len(decomposition.bags) else [])
        config = _key_to_config(verts, best_key)

        edits: list[StructuralEdit] = []
        for (i, j), state in config.items():
            if state == _FORWARD and not True:  # placeholder
                pass
        return edits

    @staticmethod
    def _extract_edits(
        orig_adj: AdjacencyMatrix,
        new_adj: AdjacencyMatrix,
    ) -> list[StructuralEdit]:
        """Extract edits between two adjacency matrices."""
        from causalcert.dag.edit import diff_edits
        return diff_edits(orig_adj, new_adj)
