"""
taintflow.utils.graph_utils – Graph algorithm utilities for DAG operations.

Provides a lightweight ``DiGraph`` adjacency-list graph plus standalone
implementations of classic graph algorithms (topological sort, SCC, dominators,
transitive closure / reduction, bridges, cut vertices, etc.) used by the
TaintFlow PI-DAG analysis engine.
"""

from __future__ import annotations

import collections
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    FrozenSet,
    Generic,
    Hashable,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
)

__all__ = [
    "DiGraph",
    "topological_sort",
    "reverse_postorder",
    "strongly_connected_components",
    "transitive_closure",
    "transitive_reduction",
    "all_paths",
    "shortest_path",
    "longest_path_dag",
    "dominators",
    "post_dominators",
    "find_back_edges",
    "condensation",
    "cut_vertices",
    "bridges",
]

N = TypeVar("N", bound=Hashable)

# ---------------------------------------------------------------------------
# DiGraph – typed adjacency-list directed graph
# ---------------------------------------------------------------------------


class DiGraph(Generic[N]):
    """Directed graph with typed nodes and optional edge data.

    Nodes can be any hashable type.  Each edge may carry an arbitrary
    Python object as *data*.
    """

    __slots__ = ("_adj", "_pred", "_node_data")

    def __init__(self) -> None:
        self._adj: Dict[N, Dict[N, Any]] = {}  # node → {successor: data}
        self._pred: Dict[N, Dict[N, Any]] = {}  # node → {predecessor: data}
        self._node_data: Dict[N, Any] = {}

    # -- construction -------------------------------------------------------

    def add_node(self, node: N, **data: Any) -> None:
        """Add *node* (idempotent).  Extra kwargs stored as node data."""
        if node not in self._adj:
            self._adj[node] = {}
            self._pred[node] = {}
        if data:
            self._node_data[node] = data

    def add_edge(self, src: N, dst: N, data: Any = None) -> None:
        """Add a directed edge *src* → *dst* with optional *data*."""
        self.add_node(src)
        self.add_node(dst)
        self._adj[src][dst] = data
        self._pred[dst][src] = data

    def remove_node(self, node: N) -> None:
        """Remove *node* and all incident edges."""
        if node not in self._adj:
            return
        # Remove outgoing edges
        for succ in list(self._adj[node]):
            del self._pred[succ][node]
        del self._adj[node]
        # Remove incoming edges
        for pred_node in list(self._pred.get(node, {})):
            if pred_node in self._adj:
                self._adj[pred_node].pop(node, None)
        self._pred.pop(node, None)
        self._node_data.pop(node, None)

    def remove_edge(self, src: N, dst: N) -> None:
        """Remove the edge *src* → *dst*."""
        if src in self._adj and dst in self._adj[src]:
            del self._adj[src][dst]
        if dst in self._pred and src in self._pred[dst]:
            del self._pred[dst][src]

    # -- queries ------------------------------------------------------------

    @property
    def nodes(self) -> List[N]:
        """Return a list of all nodes."""
        return list(self._adj)

    @property
    def edges(self) -> List[Tuple[N, N, Any]]:
        """Return ``[(src, dst, data), …]`` for all edges."""
        result: List[Tuple[N, N, Any]] = []
        for src, succs in self._adj.items():
            for dst, d in succs.items():
                result.append((src, dst, d))
        return result

    def has_node(self, node: N) -> bool:
        return node in self._adj

    def has_edge(self, src: N, dst: N) -> bool:
        return src in self._adj and dst in self._adj[src]

    def successors(self, node: N) -> List[N]:
        """Direct successors of *node*."""
        return list(self._adj.get(node, {}))

    def predecessors(self, node: N) -> List[N]:
        """Direct predecessors of *node*."""
        return list(self._pred.get(node, {}))

    def in_degree(self, node: N) -> int:
        return len(self._pred.get(node, {}))

    def out_degree(self, node: N) -> int:
        return len(self._adj.get(node, {}))

    def node_data(self, node: N) -> Any:
        return self._node_data.get(node)

    def edge_data(self, src: N, dst: N) -> Any:
        return self._adj.get(src, {}).get(dst)

    def n_nodes(self) -> int:
        return len(self._adj)

    def n_edges(self) -> int:
        return sum(len(s) for s in self._adj.values())

    # -- derived views ------------------------------------------------------

    def sources(self) -> List[N]:
        """Nodes with in-degree 0."""
        return [n for n in self._adj if not self._pred.get(n)]

    def sinks(self) -> List[N]:
        """Nodes with out-degree 0."""
        return [n for n in self._adj if not self._adj[n]]

    def is_dag(self) -> bool:
        """Return True if the graph is acyclic."""
        try:
            topological_sort(self.nodes, [(s, d) for s, d, _ in self.edges])
            return True
        except ValueError:
            return False

    def has_path(self, src: N, dst: N) -> bool:
        """BFS reachability check."""
        if src not in self._adj or dst not in self._adj:
            return False
        visited: Set[N] = set()
        queue: Deque[N] = collections.deque([src])
        while queue:
            cur = queue.popleft()
            if cur == dst:
                return True
            if cur in visited:
                continue
            visited.add(cur)
            for s in self._adj.get(cur, {}):
                if s not in visited:
                    queue.append(s)
        return False

    # -- graph transformations ----------------------------------------------

    def reverse(self) -> "DiGraph[N]":
        """Return a new graph with all edges reversed."""
        g = DiGraph[N]()
        for n, d in self._node_data.items():
            g.add_node(n, **d)
        for n in self._adj:
            g.add_node(n)
        for src, succs in self._adj.items():
            for dst, d in succs.items():
                g.add_edge(dst, src, d)
        return g

    def subgraph(self, nodes: Iterable[N]) -> "DiGraph[N]":
        """Return the induced subgraph on *nodes*."""
        keep = set(nodes)
        g = DiGraph[N]()
        for n in keep:
            if n in self._adj:
                nd = self._node_data.get(n, {})
                if nd:
                    g.add_node(n, **nd)
                else:
                    g.add_node(n)
        for src in keep:
            for dst, d in self._adj.get(src, {}).items():
                if dst in keep:
                    g.add_edge(src, dst, d)
        return g

    # -- adjacency matrix ---------------------------------------------------

    def to_adjacency_matrix(self) -> Tuple[List[N], List[List[int]]]:
        """Return ``(node_list, matrix)`` where ``matrix[i][j]=1`` iff i→j."""
        node_list = list(self._adj)
        idx = {n: i for i, n in enumerate(node_list)}
        sz = len(node_list)
        mat = [[0] * sz for _ in range(sz)]
        for src, succs in self._adj.items():
            for dst in succs:
                mat[idx[src]][idx[dst]] = 1
        return node_list, mat

    @classmethod
    def from_adjacency_matrix(
        cls,
        node_list: List[N],
        matrix: List[List[int]],
    ) -> "DiGraph[N]":
        """Build a graph from a node list and adjacency matrix."""
        g = cls()
        for n in node_list:
            g.add_node(n)
        for i, src in enumerate(node_list):
            for j, dst in enumerate(node_list):
                if matrix[i][j]:
                    g.add_edge(src, dst)
        return g

    # -- traversals ---------------------------------------------------------

    def iter_dfs(self, start: N) -> Iterator[N]:
        """Depth-first traversal from *start*."""
        visited: Set[N] = set()
        stack: List[N] = [start]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            yield node
            for s in reversed(self.successors(node)):
                if s not in visited:
                    stack.append(s)

    def iter_bfs(self, start: N) -> Iterator[N]:
        """Breadth-first traversal from *start*."""
        visited: Set[N] = set()
        queue: Deque[N] = collections.deque([start])
        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            yield node
            for s in self.successors(node):
                if s not in visited:
                    queue.append(s)

    def connected_components(self) -> List[Set[N]]:
        """Weakly connected components (ignoring edge direction)."""
        visited: Set[N] = set()
        components: List[Set[N]] = []
        # Build undirected adjacency
        und: Dict[N, Set[N]] = {n: set() for n in self._adj}
        for src, succs in self._adj.items():
            for dst in succs:
                und[src].add(dst)
                und[dst].add(src)
        for node in self._adj:
            if node in visited:
                continue
            comp: Set[N] = set()
            queue: Deque[N] = collections.deque([node])
            while queue:
                cur = queue.popleft()
                if cur in comp:
                    continue
                comp.add(cur)
                visited.add(cur)
                for nb in und.get(cur, set()):
                    if nb not in comp:
                        queue.append(nb)
            components.append(comp)
        return components

    # -- pretty printing ----------------------------------------------------

    def __repr__(self) -> str:
        return f"DiGraph(nodes={self.n_nodes()}, edges={self.n_edges()})"

    def __contains__(self, node: N) -> bool:
        return node in self._adj


# ---------------------------------------------------------------------------
# Standalone graph algorithms
# ---------------------------------------------------------------------------


def topological_sort(
    nodes: Sequence[N],
    edges: Sequence[Tuple[N, N]],
) -> List[N]:
    """Kahn's algorithm for topological sorting.

    Parameters
    ----------
    nodes : sequence of hashable
        All nodes in the graph.
    edges : sequence of (src, dst)
        Directed edges.

    Returns
    -------
    list
        Topologically ordered nodes.

    Raises
    ------
    ValueError
        If the graph contains a cycle.
    """
    adj: Dict[N, List[N]] = {n: [] for n in nodes}
    in_deg: Dict[N, int] = {n: 0 for n in nodes}
    for src, dst in edges:
        if src not in adj:
            adj[src] = []
            in_deg[src] = 0
        if dst not in adj:
            adj[dst] = []
            in_deg[dst] = 0
        adj[src].append(dst)
        in_deg[dst] += 1

    queue: Deque[N] = collections.deque(n for n in nodes if in_deg[n] == 0)
    result: List[N] = []
    while queue:
        node = queue.popleft()
        result.append(node)
        for succ in adj[node]:
            in_deg[succ] -= 1
            if in_deg[succ] == 0:
                queue.append(succ)

    if len(result) != len(adj):
        raise ValueError("graph contains a cycle; topological sort impossible")
    return result


def reverse_postorder(
    nodes: Sequence[N],
    edges: Sequence[Tuple[N, N]],
) -> List[N]:
    """Reverse post-order traversal (useful for worklist algorithms).

    Equivalent to reversed DFS post-order.  For a DAG this is a
    topological order.

    Parameters
    ----------
    nodes : sequence
        All nodes.
    edges : sequence of (src, dst)
        Directed edges.

    Returns
    -------
    list
        Nodes in reverse post-order.
    """
    adj: Dict[N, List[N]] = {n: [] for n in nodes}
    for src, dst in edges:
        adj.setdefault(src, []).append(dst)
        adj.setdefault(dst, [])

    visited: Set[N] = set()
    post_order: List[N] = []

    def _dfs(v: N) -> None:
        stack: List[Tuple[N, int]] = [(v, 0)]
        while stack:
            node, idx = stack.pop()
            if node in visited and idx == 0:
                continue
            if idx == 0:
                visited.add(node)
            children = adj.get(node, [])
            if idx < len(children):
                stack.append((node, idx + 1))
                child = children[idx]
                if child not in visited:
                    stack.append((child, 0))
            else:
                post_order.append(node)

    for n in nodes:
        if n not in visited:
            _dfs(n)

    post_order.reverse()
    return post_order


def strongly_connected_components(
    nodes: Sequence[N],
    edges: Sequence[Tuple[N, N]],
) -> List[List[N]]:
    """Tarjan's algorithm for finding strongly connected components.

    Parameters
    ----------
    nodes : sequence
        All nodes.
    edges : sequence of (src, dst)
        Directed edges.

    Returns
    -------
    list of list
        Each inner list is an SCC, returned in reverse topological order.
    """
    adj: Dict[N, List[N]] = {n: [] for n in nodes}
    for src, dst in edges:
        adj.setdefault(src, []).append(dst)
        adj.setdefault(dst, [])

    index_counter = [0]
    indices: Dict[N, int] = {}
    lowlinks: Dict[N, int] = {}
    on_stack: Set[N] = set()
    stack: List[N] = []
    result: List[List[N]] = []

    def _strongconnect(v: N) -> None:
        # iterative Tarjan's
        call_stack: List[Tuple[N, int]] = [(v, 0)]
        indices[v] = lowlinks[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack.add(v)

        while call_stack:
            node, ci = call_stack.pop()
            children = adj.get(node, [])
            recurse = False
            for i in range(ci, len(children)):
                w = children[i]
                if w not in indices:
                    indices[w] = lowlinks[w] = index_counter[0]
                    index_counter[0] += 1
                    stack.append(w)
                    on_stack.add(w)
                    call_stack.append((node, i + 1))
                    call_stack.append((w, 0))
                    recurse = True
                    break
                elif w in on_stack:
                    lowlinks[node] = min(lowlinks[node], indices[w])

            if recurse:
                continue

            # post-processing: update parent lowlink
            if call_stack:
                parent_node = call_stack[-1][0]
                lowlinks[parent_node] = min(lowlinks[parent_node], lowlinks[node])

            if lowlinks[node] == indices[node]:
                scc: List[N] = []
                while True:
                    w = stack.pop()
                    on_stack.discard(w)
                    scc.append(w)
                    if w == node:
                        break
                result.append(scc)

    for n in nodes:
        if n not in indices:
            _strongconnect(n)

    return result


def transitive_closure(
    nodes: Sequence[N],
    edges: Sequence[Tuple[N, N]],
) -> List[Tuple[N, N]]:
    """Floyd-Warshall transitive closure.

    Parameters
    ----------
    nodes : sequence
        All nodes.
    edges : sequence of (src, dst)
        Directed edges.

    Returns
    -------
    list of (src, dst)
        All pairs (u, v) such that v is reachable from u.
    """
    node_list = list(nodes)
    idx = {n: i for i, n in enumerate(node_list)}
    sz = len(node_list)
    # Reachability matrix
    reach = [[False] * sz for _ in range(sz)]
    for i in range(sz):
        reach[i][i] = True
    for src, dst in edges:
        if src in idx and dst in idx:
            reach[idx[src]][idx[dst]] = True

    for k in range(sz):
        for i in range(sz):
            if not reach[i][k]:
                continue
            for j in range(sz):
                if reach[k][j]:
                    reach[i][j] = True

    result: List[Tuple[N, N]] = []
    for i in range(sz):
        for j in range(sz):
            if i != j and reach[i][j]:
                result.append((node_list[i], node_list[j]))
    return result


def transitive_reduction(
    nodes: Sequence[N],
    edges: Sequence[Tuple[N, N]],
) -> List[Tuple[N, N]]:
    """Compute the transitive reduction (minimum equivalent edge set).

    Only meaningful for DAGs; for general graphs the result is the
    reduction of the DAG of SCCs.

    Parameters
    ----------
    nodes : sequence
        All nodes.
    edges : sequence of (src, dst)
        Directed edges.

    Returns
    -------
    list of (src, dst)
        Minimal edge set with the same reachability.
    """
    edge_set = set(edges)
    node_list = list(nodes)
    idx = {n: i for i, n in enumerate(node_list)}
    sz = len(node_list)

    # Build reachability (not via direct edge)
    adj: Dict[N, Set[N]] = {n: set() for n in nodes}
    for src, dst in edges:
        adj.setdefault(src, set()).add(dst)

    # For each edge (u, v), remove it if v is reachable from u via other paths
    to_remove: Set[Tuple[N, N]] = set()
    for src, dst in edges:
        # BFS from src, not using edge (src, dst) directly
        visited: Set[N] = set()
        queue: Deque[N] = collections.deque()
        for nb in adj.get(src, set()):
            if nb != dst:
                queue.append(nb)
        while queue:
            cur = queue.popleft()
            if cur in visited:
                continue
            visited.add(cur)
            if cur == dst:
                to_remove.add((src, dst))
                break
            for nb in adj.get(cur, set()):
                if nb not in visited:
                    queue.append(nb)

    return [e for e in edges if e not in to_remove]


def all_paths(
    source: N,
    target: N,
    graph: Mapping[N, Sequence[N]],
    max_paths: int = 10000,
) -> List[List[N]]:
    """Enumerate all simple paths from *source* to *target*.

    Parameters
    ----------
    source, target : hashable
        Start and end nodes.
    graph : mapping node → list of successors
        Adjacency list representation.
    max_paths : int
        Safety limit to avoid combinatorial explosion.

    Returns
    -------
    list of list
        Each inner list is a path from source to target.
    """
    results: List[List[N]] = []
    stack: List[Tuple[N, List[N], Set[N]]] = [(source, [source], {source})]
    while stack and len(results) < max_paths:
        node, path, visited = stack.pop()
        if node == target:
            results.append(list(path))
            continue
        for nb in graph.get(node, []):
            if nb not in visited:
                new_visited = visited | {nb}
                stack.append((nb, path + [nb], new_visited))
    return results


def shortest_path(
    source: N,
    target: N,
    graph: Mapping[N, Sequence[N]],
) -> Optional[List[N]]:
    """BFS shortest path from *source* to *target*.

    Returns
    -------
    list or None
        Shortest path as list of nodes, or None if unreachable.
    """
    if source == target:
        return [source]
    visited: Set[N] = {source}
    queue: Deque[Tuple[N, List[N]]] = collections.deque([(source, [source])])
    while queue:
        node, path = queue.popleft()
        for nb in graph.get(node, []):
            if nb == target:
                return path + [nb]
            if nb not in visited:
                visited.add(nb)
                queue.append((nb, path + [nb]))
    return None


def longest_path_dag(
    source: N,
    target: N,
    graph: Mapping[N, Sequence[N]],
) -> Optional[List[N]]:
    """Longest path in a DAG from *source* to *target*.

    Uses dynamic programming over a topological order.

    Returns
    -------
    list or None
        Longest path, or None if *target* is not reachable.
    """
    # Collect all nodes reachable from source
    all_nodes: Set[N] = set()
    queue: Deque[N] = collections.deque([source])
    while queue:
        n = queue.popleft()
        if n in all_nodes:
            continue
        all_nodes.add(n)
        for nb in graph.get(n, []):
            if nb not in all_nodes:
                queue.append(nb)

    if target not in all_nodes:
        return None

    # Build edge list for topo sort
    edge_list: List[Tuple[N, N]] = []
    for n in all_nodes:
        for nb in graph.get(n, []):
            if nb in all_nodes:
                edge_list.append((n, nb))

    try:
        topo = topological_sort(list(all_nodes), edge_list)
    except ValueError:
        return None  # cycle detected

    dist: Dict[N, int] = {n: -1 for n in all_nodes}
    prev: Dict[N, Optional[N]] = {n: None for n in all_nodes}
    dist[source] = 0

    for n in topo:
        if dist[n] == -1:
            continue
        for nb in graph.get(n, []):
            if nb in all_nodes and dist[n] + 1 > dist[nb]:
                dist[nb] = dist[n] + 1
                prev[nb] = n

    if dist[target] == -1:
        return None

    # Reconstruct path
    path: List[N] = []
    cur: Optional[N] = target
    while cur is not None:
        path.append(cur)
        cur = prev.get(cur)
    path.reverse()
    return path


# ---------------------------------------------------------------------------
# Dominator computation
# ---------------------------------------------------------------------------


def dominators(
    entry: N,
    graph: Mapping[N, Sequence[N]],
) -> Dict[N, N]:
    """Compute the immediate dominator tree using the iterative algorithm.

    Parameters
    ----------
    entry : hashable
        Entry node.
    graph : mapping node → list of successors
        Adjacency list.

    Returns
    -------
    dict
        Mapping from each node to its immediate dominator.
        The entry node maps to itself.
    """
    # Collect all reachable nodes via BFS
    reachable: List[N] = []
    visited: Set[N] = set()
    q: Deque[N] = collections.deque([entry])
    while q:
        n = q.popleft()
        if n in visited:
            continue
        visited.add(n)
        reachable.append(n)
        for s in graph.get(n, []):
            if s not in visited:
                q.append(s)

    # Build predecessor map
    pred_map: Dict[N, List[N]] = {n: [] for n in reachable}
    for n in reachable:
        for s in graph.get(n, []):
            if s in pred_map:
                pred_map[s].append(n)

    # RPO numbering
    rpo: List[N] = []
    rpo_visited: Set[N] = set()

    def _dfs_post(v: N) -> None:
        stack: List[Tuple[N, int]] = [(v, 0)]
        while stack:
            node, idx = stack.pop()
            if node in rpo_visited and idx == 0:
                continue
            if idx == 0:
                rpo_visited.add(node)
            children = list(graph.get(node, []))
            children = [c for c in children if c in pred_map]
            if idx < len(children):
                stack.append((node, idx + 1))
                child = children[idx]
                if child not in rpo_visited:
                    stack.append((child, 0))
            else:
                rpo.append(node)

    _dfs_post(entry)
    rpo.reverse()

    rpo_idx: Dict[N, int] = {n: i for i, n in enumerate(rpo)}

    idom: Dict[N, Optional[N]] = {n: None for n in reachable}
    idom[entry] = entry

    def _intersect(b1: N, b2: N) -> N:
        finger1, finger2 = b1, b2
        while finger1 != finger2:
            while rpo_idx.get(finger1, len(rpo)) > rpo_idx.get(finger2, len(rpo)):
                d = idom.get(finger1)
                if d is None or d == finger1:
                    break
                finger1 = d
            while rpo_idx.get(finger2, len(rpo)) > rpo_idx.get(finger1, len(rpo)):
                d = idom.get(finger2)
                if d is None or d == finger2:
                    break
                finger2 = d
        return finger1

    changed = True
    while changed:
        changed = False
        for n in rpo:
            if n == entry:
                continue
            preds = [p for p in pred_map[n] if idom.get(p) is not None]
            if not preds:
                continue
            new_idom = preds[0]
            for p in preds[1:]:
                new_idom = _intersect(new_idom, p)
            if idom[n] != new_idom:
                idom[n] = new_idom
                changed = True

    return {n: d for n, d in idom.items() if d is not None}


def post_dominators(
    exits: Sequence[N],
    graph: Mapping[N, Sequence[N]],
) -> Dict[N, N]:
    """Compute the immediate post-dominator tree.

    Parameters
    ----------
    exits : sequence
        Exit nodes (sinks).
    graph : mapping node → list of successors
        Forward adjacency list.

    Returns
    -------
    dict
        Mapping from each node to its immediate post-dominator.
    """
    # Build reverse graph
    all_nodes: Set[N] = set()
    for n in graph:
        all_nodes.add(n)
        for s in graph[n]:
            all_nodes.add(s)

    rev: Dict[N, List[N]] = {n: [] for n in all_nodes}
    for n in graph:
        for s in graph[n]:
            rev[s].append(n)

    # Add a virtual exit node
    virtual_exit: Any = "__post_dom_virtual_exit__"
    rev[virtual_exit] = list(exits)
    for e in exits:
        # virtual_exit is a successor in the reversed graph
        pass

    result = dominators(virtual_exit, rev)
    # Remove virtual exit from results
    cleaned: Dict[N, N] = {}
    for n, d in result.items():
        if n == virtual_exit:
            continue
        if d == virtual_exit:
            cleaned[n] = n  # no post-dominator → self
        else:
            cleaned[n] = d
    return cleaned


# ---------------------------------------------------------------------------
# Cycle / back-edge detection
# ---------------------------------------------------------------------------


def find_back_edges(
    graph: Mapping[N, Sequence[N]],
) -> List[Tuple[N, N]]:
    """Detect back edges (cycle-causing edges) via DFS classification.

    Parameters
    ----------
    graph : mapping node → list of successors

    Returns
    -------
    list of (src, dst)
        Edges where *dst* is an ancestor of *src* in the DFS tree.
    """
    WHITE, GRAY, BLACK = 0, 1, 2
    color: Dict[N, int] = {n: WHITE for n in graph}
    back_edges: List[Tuple[N, N]] = []

    for start in list(graph):
        if color.get(start, WHITE) != WHITE:
            continue
        stack: List[Tuple[N, int]] = [(start, 0)]
        color[start] = GRAY
        while stack:
            node, idx = stack.pop()
            children = list(graph.get(node, []))
            if idx < len(children):
                stack.append((node, idx + 1))
                child = children[idx]
                c = color.get(child, WHITE)
                if c == WHITE:
                    color[child] = GRAY
                    stack.append((child, 0))
                elif c == GRAY:
                    back_edges.append((node, child))
            else:
                color[node] = BLACK

    return back_edges


# ---------------------------------------------------------------------------
# Condensation (DAG of SCCs)
# ---------------------------------------------------------------------------


def condensation(
    graph: Mapping[N, Sequence[N]],
    sccs: Optional[Sequence[Sequence[N]]] = None,
) -> Tuple[DiGraph[int], Dict[N, int]]:
    """Build the condensation DAG from SCCs.

    Parameters
    ----------
    graph : mapping node → list of successors
    sccs : optional precomputed SCCs

    Returns
    -------
    (dag, node_to_scc_id)
        *dag* is a ``DiGraph[int]`` where each node is an SCC index.
        *node_to_scc_id* maps original nodes to SCC indices.
    """
    all_nodes = list(graph)
    edge_list = [(s, d) for s in graph for d in graph[s]]

    if sccs is None:
        sccs = strongly_connected_components(all_nodes, edge_list)

    node_to_id: Dict[N, int] = {}
    for idx, scc in enumerate(sccs):
        for n in scc:
            node_to_id[n] = idx

    dag: DiGraph[int] = DiGraph()
    for idx in range(len(sccs)):
        dag.add_node(idx)

    seen_edges: Set[Tuple[int, int]] = set()
    for src in graph:
        for dst in graph[src]:
            a, b = node_to_id[src], node_to_id[dst]
            if a != b and (a, b) not in seen_edges:
                dag.add_edge(a, b)
                seen_edges.add((a, b))

    return dag, node_to_id


# ---------------------------------------------------------------------------
# Cut vertices & bridges (for undirected interpretation)
# ---------------------------------------------------------------------------


def _build_undirected(
    graph: Mapping[N, Sequence[N]],
) -> Dict[N, Set[N]]:
    """Convert directed adjacency to undirected."""
    und: Dict[N, Set[N]] = {}
    for n in graph:
        und.setdefault(n, set())
        for s in graph[n]:
            und.setdefault(s, set())
            und[n].add(s)
            und[s].add(n)
    return und


def cut_vertices(
    graph: Mapping[N, Sequence[N]],
) -> List[N]:
    """Find articulation points (cut vertices) in the undirected view.

    Parameters
    ----------
    graph : mapping node → list of successors

    Returns
    -------
    list of nodes
        Articulation points.
    """
    und = _build_undirected(graph)
    nodes = list(und)
    if not nodes:
        return []

    disc: Dict[N, int] = {}
    low: Dict[N, int] = {}
    parent: Dict[N, Optional[N]] = {}
    ap: Set[N] = set()
    timer = [0]

    for start in nodes:
        if start in disc:
            continue
        # Iterative DFS
        stack: List[Tuple[N, Iterator[N], bool]] = [
            (start, iter(sorted(und.get(start, set()))), True)
        ]
        parent[start] = None
        disc[start] = low[start] = timer[0]
        timer[0] += 1
        child_count: Dict[N, int] = {start: 0}

        while stack:
            node, neighbors, is_first = stack[-1]
            try:
                nb = next(neighbors)
                if nb not in disc:
                    parent[nb] = node
                    disc[nb] = low[nb] = timer[0]
                    timer[0] += 1
                    child_count[nb] = 0
                    stack.append((nb, iter(sorted(und.get(nb, set()))), True))
                elif nb != parent.get(node):
                    low[node] = min(low[node], disc[nb])
            except StopIteration:
                stack.pop()
                if stack:
                    par = stack[-1][0]
                    low[par] = min(low[par], low[node])
                    child_count[par] = child_count.get(par, 0) + 1
                    # Check articulation point conditions
                    if parent[par] is not None and low[node] >= disc[par]:
                        ap.add(par)
                    if parent[par] is None and child_count[par] > 1:
                        ap.add(par)

    return list(ap)


def bridges(
    graph: Mapping[N, Sequence[N]],
) -> List[Tuple[N, N]]:
    """Find bridge edges in the undirected view.

    Parameters
    ----------
    graph : mapping node → list of successors

    Returns
    -------
    list of (u, v)
        Bridge edges.
    """
    und = _build_undirected(graph)
    nodes = list(und)
    if not nodes:
        return []

    disc: Dict[N, int] = {}
    low: Dict[N, int] = {}
    parent: Dict[N, Optional[N]] = {}
    bridge_list: List[Tuple[N, N]] = []
    timer = [0]

    for start in nodes:
        if start in disc:
            continue
        parent[start] = None
        disc[start] = low[start] = timer[0]
        timer[0] += 1
        stack: List[Tuple[N, Iterator[N]]] = [
            (start, iter(sorted(und.get(start, set()))))
        ]

        while stack:
            node, neighbors = stack[-1]
            try:
                nb = next(neighbors)
                if nb not in disc:
                    parent[nb] = node
                    disc[nb] = low[nb] = timer[0]
                    timer[0] += 1
                    stack.append((nb, iter(sorted(und.get(nb, set())))))
                elif nb != parent.get(node):
                    low[node] = min(low[node], disc[nb])
            except StopIteration:
                stack.pop()
                if stack:
                    par = stack[-1][0]
                    low[par] = min(low[par], low[node])
                    if low[node] > disc[par]:
                        bridge_list.append((par, node))

    return bridge_list


# ---------------------------------------------------------------------------
# Depth / level computation
# ---------------------------------------------------------------------------


def dag_depth(
    entry: N,
    graph: Mapping[N, Sequence[N]],
) -> Dict[N, int]:
    """Compute depth (longest path from *entry*) for each reachable node.

    Parameters
    ----------
    entry : hashable
        Source node.
    graph : mapping node → list of successors

    Returns
    -------
    dict
        Node → depth.
    """
    depth: Dict[N, int] = {entry: 0}
    queue: Deque[N] = collections.deque([entry])
    visited: Set[N] = set()
    while queue:
        n = queue.popleft()
        if n in visited:
            # Update depth to longest
            pass
        visited.add(n)
        for s in graph.get(n, []):
            new_d = depth[n] + 1
            if s not in depth or new_d > depth[s]:
                depth[s] = new_d
            queue.append(s)
    return depth


def dag_levels(
    graph: Mapping[N, Sequence[N]],
) -> List[List[N]]:
    """Partition DAG nodes into levels for layered visualization.

    Level 0 contains sources, level k contains nodes whose longest
    input path has length k.

    Parameters
    ----------
    graph : mapping node → list of successors

    Returns
    -------
    list of list
        ``result[k]`` contains all nodes at level k.
    """
    all_nodes: Set[N] = set(graph)
    for n in graph:
        for s in graph[n]:
            all_nodes.add(s)

    # Find in-degree
    in_deg: Dict[N, int] = {n: 0 for n in all_nodes}
    adj: Dict[N, List[N]] = {n: [] for n in all_nodes}
    for n in graph:
        for s in graph[n]:
            adj[n].append(s)
            in_deg[s] = in_deg.get(s, 0) + 1

    # BFS by levels
    current = [n for n in all_nodes if in_deg[n] == 0]
    levels: List[List[N]] = []
    remaining = dict(in_deg)

    while current:
        levels.append(current)
        next_level: List[N] = []
        for n in current:
            for s in adj.get(n, []):
                remaining[s] -= 1
                if remaining[s] == 0:
                    next_level.append(s)
        current = next_level

    return levels
