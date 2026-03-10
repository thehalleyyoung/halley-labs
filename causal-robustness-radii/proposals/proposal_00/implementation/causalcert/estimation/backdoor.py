"""
Back-door criterion checking and adjustment set enumeration.

Implements the graphical back-door criterion (Pearl, 1993) for identifying
valid adjustment sets and enumerates all minimal adjustment sets.
"""

from __future__ import annotations

from collections import deque
from itertools import combinations
from typing import Sequence

import numpy as np

from causalcert.types import AdjacencyMatrix, NodeId, NodeSet


# ---------------------------------------------------------------------------
# Internal graph helpers  (self-contained so we don't depend on dag stubs)
# ---------------------------------------------------------------------------


def _children(adj: np.ndarray, v: int) -> set[int]:
    """Return children of *v* in adjacency matrix."""
    return set(int(c) for c in np.nonzero(adj[v])[0])


def _parents(adj: np.ndarray, v: int) -> set[int]:
    """Return parents of *v* in adjacency matrix."""
    return set(int(p) for p in np.nonzero(adj[:, v])[0])


def _descendants(adj: np.ndarray, sources: set[int]) -> set[int]:
    """Return the descendant set of *sources* (inclusive) via BFS."""
    visited: set[int] = set()
    queue = deque(sources)
    while queue:
        v = queue.popleft()
        if v in visited:
            continue
        visited.add(v)
        for c in _children(adj, v):
            if c not in visited:
                queue.append(c)
    return visited


def _ancestors(adj: np.ndarray, targets: set[int]) -> set[int]:
    """Return the ancestor set of *targets* (inclusive) via reverse BFS."""
    visited: set[int] = set()
    queue = deque(targets)
    while queue:
        v = queue.popleft()
        if v in visited:
            continue
        visited.add(v)
        for p in _parents(adj, v):
            if p not in visited:
                queue.append(p)
    return visited


def _topological_order(adj: np.ndarray) -> list[int]:
    """Kahn's algorithm topological sort."""
    n = adj.shape[0]
    in_deg = adj.sum(axis=0).astype(int)
    queue = deque(i for i in range(n) if in_deg[i] == 0)
    order: list[int] = []
    while queue:
        v = queue.popleft()
        order.append(v)
        for c in range(n):
            if adj[v, c]:
                in_deg[c] -= 1
                if in_deg[c] == 0:
                    queue.append(c)
    return order


def _is_d_separated(
    adj: np.ndarray,
    x: int,
    y: int,
    conditioning: set[int],
) -> bool:
    """Bayes-Ball algorithm for d-separation.

    Returns True if x ⊥_d y | conditioning in the DAG given by *adj*.
    """
    n = adj.shape[0]
    # Ancestors of conditioning set (for collider activation)
    cond_ancestors = _ancestors(adj, conditioning)

    # States: (node, "up"/"down") representing direction of traversal
    # "up" = arrived at node from a child, "down" = arrived from a parent
    visited: set[tuple[int, str]] = set()
    queue: deque[tuple[int, str]] = deque()

    # Start: we're looking at x's neighbours
    # From x, we can go "up" to x's parents and "down" to x's children
    queue.append((x, "up"))
    queue.append((x, "down"))

    while queue:
        node, direction = queue.popleft()
        if (node, direction) in visited:
            continue
        visited.add((node, direction))

        if node == y:
            return False  # d-connected

        if direction == "up" and node not in conditioning:
            # Arrived from child — if not conditioned, can go up to parents
            # and down to other children
            for p in _parents(adj, node):
                queue.append((p, "up"))
            for c in _children(adj, node):
                queue.append((c, "down"))

        elif direction == "down":
            # Arrived from parent
            if node not in conditioning:
                # Non-collider that's not conditioned — pass through to children
                for c in _children(adj, node):
                    queue.append((c, "down"))
            if node in cond_ancestors:
                # Collider or descendant of collider that is
                # conditioned/has conditioned descendant — activate
                for p in _parents(adj, node):
                    queue.append((p, "up"))

    return True  # No active path found


def _backdoor_paths_blocked(
    adj: np.ndarray,
    treatment: int,
    outcome: int,
    conditioning: set[int],
) -> bool:
    """Check that all back-door paths from treatment to outcome are blocked.

    A back-door path is any path from treatment to outcome that starts
    with an arrow *into* treatment.  We check this by removing all edges
    *out of* treatment and testing d-separation.
    """
    adj_mod = adj.copy()
    adj_mod[treatment, :] = 0  # Remove all edges out of treatment
    return _is_d_separated(adj_mod, treatment, outcome, conditioning)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def satisfies_backdoor(
    adj: AdjacencyMatrix,
    treatment: NodeId,
    outcome: NodeId,
    adjustment_set: NodeSet,
) -> bool:
    """Check whether *adjustment_set* satisfies the back-door criterion.

    The back-door criterion requires that the adjustment set:
    1. Contains no descendant of the treatment.
    2. Blocks all back-door paths (non-causal paths) between treatment
       and outcome.

    Parameters
    ----------
    adj : AdjacencyMatrix
        DAG adjacency matrix.
    treatment : NodeId
        Treatment variable.
    outcome : NodeId
        Outcome variable.
    adjustment_set : NodeSet
        Candidate adjustment set.

    Returns
    -------
    bool
        ``True`` if the back-door criterion is satisfied.
    """
    adj = np.asarray(adj, dtype=np.int8)
    s = set(adjustment_set)

    # Condition 1: S contains no descendant of X
    # (descendants includes X itself; we exclude X from check since
    #  adjustment set shouldn't contain treatment anyway)
    desc_x = _descendants(adj, {treatment})
    if s & desc_x:
        return False

    # Condition 2: S blocks all back-door paths
    return _backdoor_paths_blocked(adj, treatment, outcome, s)


def enumerate_adjustment_sets(
    adj: AdjacencyMatrix,
    treatment: NodeId,
    outcome: NodeId,
    *,
    minimal: bool = True,
    max_size: int | None = None,
) -> list[NodeSet]:
    """Enumerate all valid adjustment sets.

    Parameters
    ----------
    adj : AdjacencyMatrix
        DAG adjacency matrix.
    treatment : NodeId
        Treatment variable.
    outcome : NodeId
        Outcome variable.
    minimal : bool
        If ``True``, only return inclusion-minimal sets.
    max_size : int | None
        Maximum adjustment set size.

    Returns
    -------
    list[NodeSet]
        Valid adjustment sets.
    """
    adj = np.asarray(adj, dtype=np.int8)
    n = adj.shape[0]

    # Candidate nodes: all nodes except treatment, outcome, and descendants of treatment
    desc_x = _descendants(adj, {treatment})
    candidates = sorted(
        i for i in range(n)
        if i != treatment and i != outcome and i not in desc_x
    )

    if max_size is None:
        max_size = len(candidates)
    else:
        max_size = min(max_size, len(candidates))

    valid_sets: list[NodeSet] = []

    # Check empty set first
    if satisfies_backdoor(adj, treatment, outcome, frozenset()):
        valid_sets.append(frozenset())

    # Enumerate subsets by increasing size
    for size in range(1, max_size + 1):
        for combo in combinations(candidates, size):
            s = frozenset(combo)
            if satisfies_backdoor(adj, treatment, outcome, s):
                valid_sets.append(s)

    if minimal:
        valid_sets = _filter_minimal(valid_sets)

    return valid_sets


def _filter_minimal(sets: list[NodeSet]) -> list[NodeSet]:
    """Keep only inclusion-minimal sets."""
    # Sort by size
    sets_sorted = sorted(sets, key=len)
    minimal: list[NodeSet] = []
    for s in sets_sorted:
        if not any(m <= s for m in minimal):
            minimal.append(s)
    return minimal


def find_minimum_adjustment_set(
    adj: AdjacencyMatrix,
    treatment: NodeId,
    outcome: NodeId,
) -> NodeSet:
    """Find a smallest valid adjustment set by cardinality.

    Parameters
    ----------
    adj : AdjacencyMatrix
        DAG adjacency matrix.
    treatment : NodeId
        Treatment variable.
    outcome : NodeId
        Outcome variable.

    Returns
    -------
    NodeSet
        A minimum-cardinality valid adjustment set.

    Raises
    ------
    NoValidAdjustmentSetError
        If no valid adjustment set exists.
    """
    from causalcert.exceptions import NoValidAdjustmentSetError

    adj = np.asarray(adj, dtype=np.int8)
    n = adj.shape[0]
    desc_x = _descendants(adj, {treatment})
    candidates = sorted(
        i for i in range(n)
        if i != treatment and i != outcome and i not in desc_x
    )

    # Check empty set
    if satisfies_backdoor(adj, treatment, outcome, frozenset()):
        return frozenset()

    # BFS over set sizes
    for size in range(1, len(candidates) + 1):
        for combo in combinations(candidates, size):
            s = frozenset(combo)
            if satisfies_backdoor(adj, treatment, outcome, s):
                return s

    raise NoValidAdjustmentSetError(treatment, outcome)


def find_optimal_adjustment_set_backdoor(
    adj: AdjacencyMatrix,
    treatment: NodeId,
    outcome: NodeId,
) -> NodeSet:
    """Find the O-set (Henckel, Perrier & Maathuis, 2022) via the back-door.

    The optimal adjustment set O is the union of:
    - parents of the outcome that are not descendants of treatment, and
    - parents of treatment that are not descendants of treatment.

    More precisely, O = pa(cn(treatment, outcome)) ∩ non-desc(treatment) ∩ an({treatment,outcome}).

    For simplicity we use the Tian-Pearl approximation: O = pa(Y) \\ desc(X),
    augmented with pa(X) \\ desc(X) if needed for validity.

    Parameters
    ----------
    adj : AdjacencyMatrix
        DAG adjacency matrix.
    treatment : NodeId
        Treatment variable.
    outcome : NodeId
        Outcome variable.

    Returns
    -------
    NodeSet
        Optimal adjustment set.

    Raises
    ------
    NoValidAdjustmentSetError
        If no valid adjustment set exists.
    """
    from causalcert.exceptions import NoValidAdjustmentSetError

    adj_arr = np.asarray(adj, dtype=np.int8)
    desc_x = _descendants(adj_arr, {treatment})

    # O-set construction (Henckel et al. 2022, Theorem 4):
    # O = (an({X, Y} ∪ pa(causal_path_nodes)) \ desc(X)) ∩ non_forbidden
    # Simplified: start with pa(Y) \ desc(X)
    pa_y = _parents(adj_arr, outcome)
    o_set = pa_y - desc_x - {treatment, outcome}

    # Check validity
    if satisfies_backdoor(adj_arr, treatment, outcome, frozenset(o_set)):
        return frozenset(o_set)

    # Augment with parents of treatment
    pa_x = _parents(adj_arr, treatment)
    o_set |= (pa_x - desc_x - {treatment, outcome})

    if satisfies_backdoor(adj_arr, treatment, outcome, frozenset(o_set)):
        return frozenset(o_set)

    # Augment further with ancestors of {X, Y} minus forbidden
    anc = _ancestors(adj_arr, {treatment, outcome})
    o_set = (anc - desc_x - {treatment, outcome})

    if satisfies_backdoor(adj_arr, treatment, outcome, frozenset(o_set)):
        return frozenset(o_set)

    # Fall back to enumeration
    sets = enumerate_adjustment_sets(adj_arr, treatment, outcome, minimal=False, max_size=None)
    if not sets:
        raise NoValidAdjustmentSetError(treatment, outcome)

    # Among valid sets pick the one with smallest asymptotic variance proxy
    # (heuristic: prefer sets with more parents of Y)
    pa_y_set = _parents(adj_arr, outcome)
    best = max(sets, key=lambda s: len(s & pa_y_set))
    return best


def check_adjustment_criterion(
    adj: AdjacencyMatrix,
    treatment: NodeId,
    outcome: NodeId,
    adjustment_set: NodeSet,
) -> bool:
    """Check the generalised adjustment criterion (Perkovic et al., 2018).

    This generalises the back-door criterion to allow adjustment for
    some descendants of X that are not on a causal path from X to Y.

    For a DAG, the generalised adjustment criterion reduces to checking:
    1. No node in S is a descendant of X on a proper causal path X → ... → Y.
    2. S blocks all non-causal paths from X to Y.

    Parameters
    ----------
    adj : AdjacencyMatrix
        DAG adjacency matrix.
    treatment : NodeId
        Treatment.
    outcome : NodeId
        Outcome.
    adjustment_set : NodeSet
        Candidate set.

    Returns
    -------
    bool
        ``True`` if the generalised adjustment criterion is satisfied.
    """
    adj_arr = np.asarray(adj, dtype=np.int8)
    s = set(adjustment_set)

    # Find nodes on proper causal paths from X to Y
    proper_causal = _nodes_on_proper_causal_paths(adj_arr, treatment, outcome)

    # Condition 1: S must not contain any node on proper causal paths
    if s & proper_causal:
        return False

    # Condition 2: S blocks all non-causal paths
    return _backdoor_paths_blocked(adj_arr, treatment, outcome, s)


def _nodes_on_proper_causal_paths(
    adj: np.ndarray,
    treatment: int,
    outcome: int,
) -> set[int]:
    """Find all nodes (excluding X) on proper directed paths from X to Y."""
    n = adj.shape[0]
    on_path: set[int] = set()

    # DFS from treatment; collect nodes that have a directed path to outcome
    desc_x = _descendants(adj, {treatment}) - {treatment}
    anc_y = _ancestors(adj, {outcome}) - {outcome}

    # Nodes on proper causal paths: descendants of X that are also ancestors of Y
    on_path = desc_x & anc_y
    return on_path


def all_backdoor_paths(
    adj: AdjacencyMatrix,
    treatment: NodeId,
    outcome: NodeId,
) -> list[list[NodeId]]:
    """Enumerate all back-door paths from treatment to outcome.

    A back-door path starts with an arrow *into* treatment.

    Parameters
    ----------
    adj : AdjacencyMatrix
        DAG adjacency matrix.
    treatment : NodeId
        Treatment variable.
    outcome : NodeId
        Outcome variable.

    Returns
    -------
    list[list[NodeId]]
        Each path is a list of node ids.
    """
    adj_arr = np.asarray(adj, dtype=np.int8)
    n = adj_arr.shape[0]
    paths: list[list[int]] = []

    def _dfs(current: int, target: int, visited: set[int], path: list[int]) -> None:
        if current == target:
            paths.append(list(path))
            return
        for nb in range(n):
            if nb in visited:
                continue
            # Can traverse in either direction in the moral/underlying graph
            if adj_arr[current, nb] or adj_arr[nb, current]:
                visited.add(nb)
                path.append(nb)
                _dfs(nb, target, visited, path)
                path.pop()
                visited.remove(nb)

    # Back-door paths start with an arrow into treatment
    parents_x = _parents(adj_arr, treatment)
    for p in parents_x:
        visited = {treatment, p}
        _dfs(p, outcome, visited, [treatment, p])

    return paths


def has_valid_adjustment_set(
    adj: AdjacencyMatrix,
    treatment: NodeId,
    outcome: NodeId,
) -> bool:
    """Check whether any valid back-door adjustment set exists.

    Parameters
    ----------
    adj : AdjacencyMatrix
        DAG adjacency matrix.
    treatment, outcome : NodeId
        Treatment and outcome.

    Returns
    -------
    bool
        ``True`` if at least one valid adjustment set exists.
    """
    adj_arr = np.asarray(adj, dtype=np.int8)
    n = adj_arr.shape[0]
    desc_x = _descendants(adj_arr, {treatment})
    candidates = sorted(
        i for i in range(n) if i != treatment and i != outcome and i not in desc_x
    )

    # Check all subsets up to full candidate set
    if satisfies_backdoor(adj_arr, treatment, outcome, frozenset()):
        return True

    for size in range(1, len(candidates) + 1):
        for combo in combinations(candidates, size):
            if satisfies_backdoor(adj_arr, treatment, outcome, frozenset(combo)):
                return True
    return False


def is_ancestor(adj: AdjacencyMatrix, u: NodeId, v: NodeId) -> bool:
    """Check whether *u* is an ancestor of *v* in the DAG."""
    adj_arr = np.asarray(adj, dtype=np.int8)
    return u in _ancestors(adj_arr, {v})


def is_descendant(adj: AdjacencyMatrix, u: NodeId, v: NodeId) -> bool:
    """Check whether *u* is a descendant of *v* in the DAG."""
    adj_arr = np.asarray(adj, dtype=np.int8)
    return u in _descendants(adj_arr, {v})
