"""
taintflow.dag.traversal -- Graph traversal algorithms for the PI-DAG.

Provides graph traversal and analysis algorithms for the Pipeline Information
DAG (PI-DAG) used by the TaintFlow leakage auditor:

* BFS and DFS traversals (forward and backward)
* Reachability analysis
* Path finding (all simple paths, shortest path, critical paths)
* Dominator and post-dominator tree computation
* Forward and backward program slicing for taint propagation
* Cycle detection and DAG validation
* Subgraph extraction (between split/merge points, by predicate)
* Topological level assignment
* Critical path analysis for leakage-potential estimation
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from taintflow.dag.pidag import PIDAG


# ===================================================================
#  BFS / DFS traversals
# ===================================================================


def bfs_forward(
    dag: "PIDAG",
    start_ids: Optional[List[str]] = None,
) -> Iterator[str]:
    """Breadth-first traversal following successor (forward) edges.

    Parameters
    ----------
    dag
        The pipeline DAG to traverse.
    start_ids
        Node IDs to begin from.  Defaults to source nodes.

    Yields
    ------
    Node IDs in BFS visitation order.
    """
    if start_ids is None:
        start_ids = [n.node_id for n in dag.sources]
    visited: Set[str] = set()
    queue: deque[str] = deque()
    for nid in start_ids:
        if nid not in visited and dag.has_node(nid):
            visited.add(nid)
            queue.append(nid)
    while queue:
        current = queue.popleft()
        yield current
        for succ in dag.successors(current):
            if succ not in visited:
                visited.add(succ)
                queue.append(succ)


def bfs_backward(
    dag: "PIDAG",
    start_ids: Optional[List[str]] = None,
) -> Iterator[str]:
    """Breadth-first traversal following predecessor (backward) edges.

    Parameters
    ----------
    dag
        The pipeline DAG to traverse.
    start_ids
        Node IDs to begin from.  Defaults to sink nodes.

    Yields
    ------
    Node IDs in reverse-BFS visitation order.
    """
    if start_ids is None:
        start_ids = [n.node_id for n in dag.sinks]
    visited: Set[str] = set()
    queue: deque[str] = deque()
    for nid in start_ids:
        if nid not in visited and dag.has_node(nid):
            visited.add(nid)
            queue.append(nid)
    while queue:
        current = queue.popleft()
        yield current
        for pred in dag.predecessors(current):
            if pred not in visited:
                visited.add(pred)
                queue.append(pred)


def dfs_forward(
    dag: "PIDAG",
    start_ids: Optional[List[str]] = None,
    order: str = "pre",
) -> Iterator[str]:
    """Depth-first traversal following successor (forward) edges.

    Parameters
    ----------
    dag
        The pipeline DAG to traverse.
    start_ids
        Node IDs to begin from.  Defaults to source nodes.
    order
        ``"pre"`` for pre-order (node visited before children),
        ``"post"`` for post-order (node visited after children).

    Yields
    ------
    Node IDs in DFS visitation order.
    """
    if start_ids is None:
        start_ids = [n.node_id for n in dag.sources]
    visited: Set[str] = set()

    def _visit(nid: str) -> Iterator[str]:
        if nid in visited:
            return
        visited.add(nid)
        if order == "pre":
            yield nid
        for succ in dag.successors(nid):
            yield from _visit(succ)
        if order == "post":
            yield nid

    for sid in start_ids:
        yield from _visit(sid)


def dfs_backward(
    dag: "PIDAG",
    start_ids: Optional[List[str]] = None,
    order: str = "pre",
) -> Iterator[str]:
    """Depth-first traversal following predecessor (backward) edges.

    Parameters
    ----------
    dag
        The pipeline DAG to traverse.
    start_ids
        Node IDs to begin from.  Defaults to sink nodes.
    order
        ``"pre"`` for pre-order, ``"post"`` for post-order.

    Yields
    ------
    Node IDs in reverse-DFS visitation order.
    """
    if start_ids is None:
        start_ids = [n.node_id for n in dag.sinks]
    visited: Set[str] = set()

    def _visit(nid: str) -> Iterator[str]:
        if nid in visited:
            return
        visited.add(nid)
        if order == "pre":
            yield nid
        for pred in dag.predecessors(nid):
            yield from _visit(pred)
        if order == "post":
            yield nid

    for sid in start_ids:
        yield from _visit(sid)


# ===================================================================
#  ReachabilityAnalyzer
# ===================================================================


class ReachabilityAnalyzer:
    """Determine reachability relationships in the PI-DAG.

    Computes which nodes can be reached from a given set of origins
    (forward reachability) or which nodes can reach a given set of
    targets (backward reachability).
    """

    def __init__(self, dag: "PIDAG") -> None:
        self._dag = dag

    def reachable_from(self, node_ids: Set[str]) -> Set[str]:
        """Return all nodes reachable forward from *node_ids*.

        The result **includes** the starting nodes themselves.
        """
        reached: Set[str] = set()
        queue: deque[str] = deque()
        for nid in node_ids:
            if self._dag.has_node(nid) and nid not in reached:
                reached.add(nid)
                queue.append(nid)
        while queue:
            current = queue.popleft()
            for succ in self._dag.successors(current):
                if succ not in reached:
                    reached.add(succ)
                    queue.append(succ)
        return reached

    def reachable_to(self, node_ids: Set[str]) -> Set[str]:
        """Return all nodes that can reach any node in *node_ids*.

        The result **includes** the target nodes themselves.
        """
        reached: Set[str] = set()
        queue: deque[str] = deque()
        for nid in node_ids:
            if self._dag.has_node(nid) and nid not in reached:
                reached.add(nid)
                queue.append(nid)
        while queue:
            current = queue.popleft()
            for pred in self._dag.predecessors(current):
                if pred not in reached:
                    reached.add(pred)
                    queue.append(pred)
        return reached

    def are_connected(self, source: str, target: str) -> bool:
        """Return ``True`` if *target* is forward-reachable from *source*."""
        if source == target:
            return True
        visited: Set[str] = {source}
        queue: deque[str] = deque([source])
        while queue:
            current = queue.popleft()
            for succ in self._dag.successors(current):
                if succ == target:
                    return True
                if succ not in visited:
                    visited.add(succ)
                    queue.append(succ)
        return False

    def reachability_matrix(self) -> Dict[str, Set[str]]:
        """Compute the full forward-reachability set for every node.

        Returns a dict mapping each ``node_id`` to the set of node IDs
        reachable from it (including itself).  Computed efficiently in
        reverse-topological order so each node inherits the already-
        computed reachability of its successors.
        """
        result: Dict[str, Set[str]] = {}
        for nid in reversed(self._dag.topological_order()):
            reachable: Set[str] = {nid}
            for succ in self._dag.successors(nid):
                reachable |= result.get(succ, {succ})
            result[nid] = reachable
        return result


# ===================================================================
#  PathFinder
# ===================================================================


class PathFinder:
    """Find paths between nodes in the PI-DAG.

    Supports enumeration of all simple paths, shortest-path queries,
    and critical-path (longest weighted path) computation.
    """

    def __init__(self, dag: "PIDAG") -> None:
        self._dag = dag

    def all_paths(
        self,
        source: str,
        target: str,
        max_depth: int = 100,
    ) -> List[List[str]]:
        """Enumerate all simple (non-repeating) paths from *source* to *target*.

        Parameters
        ----------
        source, target
            Endpoint node IDs.
        max_depth
            Maximum path length to prevent combinatorial explosion.

        Returns
        -------
        List of paths, where each path is a list of node IDs.
        """
        if not self._dag.has_node(source) or not self._dag.has_node(target):
            return []
        paths: List[List[str]] = []
        stack: List[Tuple[str, List[str], Set[str]]] = [
            (source, [source], {source}),
        ]
        while stack:
            current, path, visited = stack.pop()
            if current == target and len(path) > 1:
                paths.append(list(path))
                continue
            if len(path) > max_depth:
                continue
            for succ in self._dag.successors(current):
                if succ not in visited:
                    stack.append((succ, path + [succ], visited | {succ}))
        return paths

    def shortest_path(self, source: str, target: str) -> Optional[List[str]]:
        """Find the shortest path (fewest edges) from *source* to *target*.

        Returns ``None`` if no path exists.
        """
        if not self._dag.has_node(source) or not self._dag.has_node(target):
            return None
        if source == target:
            return [source]
        parent: Dict[str, Optional[str]] = {source: None}
        queue: deque[str] = deque([source])
        while queue:
            current = queue.popleft()
            for succ in self._dag.successors(current):
                if succ not in parent:
                    parent[succ] = current
                    if succ == target:
                        path: List[str] = []
                        node: Optional[str] = target
                        while node is not None:
                            path.append(node)
                            node = parent[node]
                        path.reverse()
                        return path
                    queue.append(succ)
        return None

    def critical_paths(
        self,
        weight_fn: Optional[Callable[[str, str], float]] = None,
        top_k: int = 5,
    ) -> List[Tuple[float, List[str]]]:
        """Find the *top_k* longest weighted paths through the DAG.

        Parameters
        ----------
        weight_fn
            ``(source_id, target_id) -> weight`` for an edge.
            Defaults to ``1.0`` per edge (longest by hop count).
        top_k
            Number of paths to return.

        Returns
        -------
        List of ``(total_weight, path)`` tuples sorted descending by weight.
        """
        if weight_fn is None:
            weight_fn = lambda s, t: 1.0

        topo = self._dag.topological_order()
        if not topo:
            return []

        dist: Dict[str, float] = {nid: 0.0 for nid in topo}
        pred: Dict[str, Optional[str]] = {nid: None for nid in topo}

        for nid in topo:
            for succ in self._dag.successors(nid):
                w = weight_fn(nid, succ)
                if dist[succ] < dist[nid] + w:
                    dist[succ] = dist[nid] + w
                    pred[succ] = nid

        ranked = sorted(
            ((nid, d) for nid, d in dist.items() if d > 0.0),
            key=lambda x: -x[1],
        )

        paths: List[Tuple[float, List[str]]] = []
        seen_ends: Set[str] = set()
        for end_nid, total_weight in ranked:
            if end_nid in seen_ends:
                continue
            seen_ends.add(end_nid)
            path: List[str] = []
            cur: Optional[str] = end_nid
            while cur is not None:
                path.append(cur)
                cur = pred[cur]
            path.reverse()
            if len(path) > 1:
                paths.append((total_weight, path))
            if len(paths) >= top_k:
                break

        return paths

    def path_exists(self, source: str, target: str) -> bool:
        """Return ``True`` if any path from *source* to *target* exists."""
        if not self._dag.has_node(source) or not self._dag.has_node(target):
            return False
        if source == target:
            return True
        visited: Set[str] = {source}
        queue: deque[str] = deque([source])
        while queue:
            current = queue.popleft()
            for succ in self._dag.successors(current):
                if succ == target:
                    return True
                if succ not in visited:
                    visited.add(succ)
                    queue.append(succ)
        return False


# ===================================================================
#  DominatorTree
# ===================================================================


@dataclass
class DominatorResult:
    """Result of dominator-tree computation.

    Attributes
    ----------
    idom
        Immediate-dominator map: ``node_id -> immediate_dominator_id``.
        The entry node maps to itself.
    dom_tree_children
        Dominator tree as child lists: ``node_id -> {dominated children}``.
    dominance_frontier
        Dominance frontier: ``node_id -> {frontier node IDs}``.
    """

    idom: Dict[str, str] = field(default_factory=dict)
    dom_tree_children: Dict[str, Set[str]] = field(
        default_factory=lambda: defaultdict(set),
    )
    dominance_frontier: Dict[str, Set[str]] = field(
        default_factory=lambda: defaultdict(set),
    )


class DominatorTree:
    """Compute dominator and post-dominator trees for the PI-DAG.

    Uses the iterative Cooper–Harvey–Kennedy algorithm, which converges
    in one pass for DAGs (no back edges).
    """

    def __init__(self, dag: "PIDAG") -> None:
        self._dag = dag

    # -- public API ----------------------------------------------------------

    def compute_dominators(
        self,
        entry: Optional[str] = None,
    ) -> DominatorResult:
        """Compute the dominator tree rooted at *entry*.

        Parameters
        ----------
        entry
            Root node ID.  Defaults to the first source node.

        Returns
        -------
        :class:`DominatorResult` with *idom*, *dom_tree_children*, and
        *dominance_frontier*.
        """
        if entry is None:
            sources = self._dag.sources
            if not sources:
                return DominatorResult()
            entry = sources[0].node_id

        topo = self._dag.topological_order()
        order_map: Dict[str, int] = {nid: i for i, nid in enumerate(topo)}

        idom: Dict[str, str] = {entry: entry}
        changed = True
        while changed:
            changed = False
            for nid in topo:
                if nid == entry:
                    continue
                preds = [p for p in self._dag.predecessors(nid) if p in idom]
                if not preds:
                    continue
                new_idom = preds[0]
                for p in preds[1:]:
                    new_idom = self._intersect(idom, order_map, new_idom, p)
                if idom.get(nid) != new_idom:
                    idom[nid] = new_idom
                    changed = True

        result = DominatorResult(idom=idom)
        for nid, dom in idom.items():
            if nid != dom:
                result.dom_tree_children[dom].add(nid)

        # Dominance frontier computation
        for nid in topo:
            preds = self._dag.predecessors(nid)
            if len(preds) >= 2:
                for p in preds:
                    runner = p
                    while runner in idom and runner != idom.get(nid, nid):
                        result.dominance_frontier[runner].add(nid)
                        if runner == idom[runner]:
                            break
                        runner = idom[runner]

        return result

    def compute_post_dominators(
        self,
        exit_node: Optional[str] = None,
    ) -> DominatorResult:
        """Compute the post-dominator tree rooted at *exit_node*.

        Parameters
        ----------
        exit_node
            Exit (sink) node ID.  Defaults to the first sink node.

        Returns
        -------
        :class:`DominatorResult` computed on the reversed graph.
        """
        if exit_node is None:
            sinks = self._dag.sinks
            if not sinks:
                return DominatorResult()
            exit_node = sinks[0].node_id

        reverse_topo = list(reversed(self._dag.topological_order()))
        order_map: Dict[str, int] = {nid: i for i, nid in enumerate(reverse_topo)}

        idom: Dict[str, str] = {exit_node: exit_node}
        changed = True
        while changed:
            changed = False
            for nid in reverse_topo:
                if nid == exit_node:
                    continue
                succs = [s for s in self._dag.successors(nid) if s in idom]
                if not succs:
                    continue
                new_idom = succs[0]
                for s in succs[1:]:
                    new_idom = self._intersect(idom, order_map, new_idom, s)
                if idom.get(nid) != new_idom:
                    idom[nid] = new_idom
                    changed = True

        result = DominatorResult(idom=idom)
        for nid, dom in idom.items():
            if nid != dom:
                result.dom_tree_children[dom].add(nid)

        for nid in reverse_topo:
            succs = self._dag.successors(nid)
            if len(succs) >= 2:
                for s in succs:
                    runner = s
                    while runner in idom and runner != idom.get(nid, nid):
                        result.dominance_frontier[runner].add(nid)
                        if runner == idom[runner]:
                            break
                        runner = idom[runner]

        return result

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _intersect(
        idom: Dict[str, str],
        order_map: Dict[str, int],
        a: str,
        b: str,
    ) -> str:
        """Find the nearest common dominator of *a* and *b*."""
        finger_a, finger_b = a, b
        while finger_a != finger_b:
            while order_map.get(finger_a, -1) > order_map.get(finger_b, -1):
                finger_a = idom.get(finger_a, finger_a)
                if finger_a == idom.get(finger_a):
                    break
            while order_map.get(finger_b, -1) > order_map.get(finger_a, -1):
                finger_b = idom.get(finger_b, finger_b)
                if finger_b == idom.get(finger_b):
                    break
            if finger_a == finger_b:
                break
            if (
                order_map.get(finger_a, -1) == order_map.get(finger_b, -1)
                and finger_a != finger_b
            ):
                break
        return finger_a


# ===================================================================
#  SlicingEngine
# ===================================================================


class SlicingEngine:
    """Forward and backward program slicing for taint propagation.

    A *backward slice* from node N contains every node whose output
    could affect the computation at N.  A *forward slice* from N
    contains every node whose input could be influenced by N.  These
    correspond directly to taint-propagation queries in the leakage
    auditor.
    """

    def __init__(self, dag: "PIDAG") -> None:
        self._dag = dag

    def forward_slice(
        self,
        node_id: str,
        edge_filter: Optional[Callable[[str, str], bool]] = None,
    ) -> Set[str]:
        """Compute the forward slice from *node_id*.

        Parameters
        ----------
        node_id
            The origin of the slice.
        edge_filter
            Optional predicate ``(source_id, target_id) -> bool``.
            Only edges for which the predicate returns ``True`` are followed.

        Returns
        -------
        Set of node IDs in the forward slice (includes *node_id*).
        """
        sliced: Set[str] = {node_id}
        queue: deque[str] = deque([node_id])
        while queue:
            current = queue.popleft()
            for succ in self._dag.successors(current):
                if succ not in sliced:
                    if edge_filter is None or edge_filter(current, succ):
                        sliced.add(succ)
                        queue.append(succ)
        return sliced

    def backward_slice(
        self,
        node_id: str,
        edge_filter: Optional[Callable[[str, str], bool]] = None,
    ) -> Set[str]:
        """Compute the backward slice from *node_id*.

        Parameters
        ----------
        node_id
            The criterion of the slice.
        edge_filter
            Optional predicate ``(source_id, target_id) -> bool``.
            Only edges for which the predicate returns ``True`` are followed.

        Returns
        -------
        Set of node IDs in the backward slice (includes *node_id*).
        """
        sliced: Set[str] = {node_id}
        queue: deque[str] = deque([node_id])
        while queue:
            current = queue.popleft()
            for pred in self._dag.predecessors(current):
                if pred not in sliced:
                    if edge_filter is None or edge_filter(pred, current):
                        sliced.add(pred)
                        queue.append(pred)
        return sliced

    def bidirectional_slice(self, node_id: str) -> Set[str]:
        """Union of forward and backward slices from *node_id*."""
        return self.forward_slice(node_id) | self.backward_slice(node_id)

    def chop(self, source: str, target: str) -> Set[str]:
        """Compute the *chop* between *source* and *target*.

        The chop is the intersection of the forward slice from *source*
        and the backward slice from *target* — i.e. the nodes that lie
        on some path from *source* to *target*.
        """
        fwd = self.forward_slice(source)
        bwd = self.backward_slice(target)
        return fwd & bwd


# ===================================================================
#  LoopDetector
# ===================================================================


@dataclass
class CycleReport:
    """Result of cycle detection.

    Attributes
    ----------
    is_acyclic
        ``True`` if no cycles were found.
    cycles
        List of cycles, each represented as a list of node IDs
        forming a closed walk.
    back_edges
        ``(source, target)`` pairs forming back edges in the DFS tree.
    """

    is_acyclic: bool = True
    cycles: List[List[str]] = field(default_factory=list)
    back_edges: List[Tuple[str, str]] = field(default_factory=list)


class LoopDetector:
    """Detect cycles in the DAG (validation: a PI-DAG must be acyclic).

    Even though a well-formed PI-DAG is acyclic by construction, this
    detector serves as a validation pass that can catch construction bugs
    or malformed pipeline graphs.
    """

    def __init__(self, dag: "PIDAG") -> None:
        self._dag = dag

    def detect_cycles(self) -> CycleReport:
        """Run cycle detection and return a :class:`CycleReport`.

        Uses a three-colour DFS (WHITE → GRAY → BLACK) to find back
        edges.  When a back edge ``(u, v)`` is found the cycle through
        the DFS-tree parent chain from *u* back to *v* is reconstructed.
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        colour: Dict[str, int] = {nid: WHITE for nid in self._dag.nodes}
        parent: Dict[str, Optional[str]] = {nid: None for nid in self._dag.nodes}
        report = CycleReport()

        def _dfs(nid: str) -> None:
            colour[nid] = GRAY
            for succ in self._dag.successors(nid):
                if colour.get(succ) == GRAY:
                    report.is_acyclic = False
                    report.back_edges.append((nid, succ))
                    cycle = self._reconstruct_cycle(parent, nid, succ)
                    report.cycles.append(cycle)
                elif colour.get(succ) == WHITE:
                    parent[succ] = nid
                    _dfs(succ)
            colour[nid] = BLACK

        for nid in self._dag.nodes:
            if colour[nid] == WHITE:
                _dfs(nid)

        return report

    def validate_acyclicity(self) -> bool:
        """Return ``True`` if the DAG contains no cycles."""
        return self.detect_cycles().is_acyclic

    @staticmethod
    def _reconstruct_cycle(
        parent: Dict[str, Optional[str]],
        back_src: str,
        back_tgt: str,
    ) -> List[str]:
        """Reconstruct the cycle from back edge ``(back_src -> back_tgt)``."""
        cycle: List[str] = [back_tgt]
        node: Optional[str] = back_src
        while node is not None and node != back_tgt:
            cycle.append(node)
            node = parent.get(node)
        cycle.append(back_tgt)
        cycle.reverse()
        return cycle


# ===================================================================
#  SubgraphExtractor
# ===================================================================


class SubgraphExtractor:
    """Extract subgraphs from the PI-DAG based on various criteria.

    Typical uses include isolating the subgraph between a train/test
    split point and a downstream merge point, or extracting all nodes
    that satisfy a user-defined predicate.
    """

    def __init__(self, dag: "PIDAG") -> None:
        self._dag = dag

    def extract_between(self, sources: Set[str], sinks: Set[str]) -> Set[str]:
        """Extract nodes on any path from *sources* to *sinks*.

        Returns the set of node IDs that lie on at least one simple
        path from a node in *sources* to a node in *sinks*.
        """
        ra = ReachabilityAnalyzer(self._dag)
        forward = ra.reachable_from(sources)
        backward = ra.reachable_to(sinks)
        return forward & backward

    def extract_by_predicate(
        self,
        predicate: Callable[[str], bool],
        include_connecting: bool = False,
    ) -> Set[str]:
        """Extract nodes satisfying *predicate*.

        Parameters
        ----------
        predicate
            ``(node_id) -> bool``; nodes returning ``True`` are included.
        include_connecting
            If ``True``, also include nodes on shortest paths between
            every pair of matching nodes so the result is connected.

        Returns
        -------
        Set of matching node IDs (plus connectors if requested).
        """
        matched: Set[str] = {nid for nid in self._dag.nodes if predicate(nid)}
        if not include_connecting or len(matched) < 2:
            return matched

        connected = set(matched)
        finder = PathFinder(self._dag)
        matched_list = sorted(matched)
        for i, src in enumerate(matched_list):
            for tgt in matched_list[i + 1 :]:
                path = finder.shortest_path(src, tgt)
                if path is not None:
                    connected.update(path)
        return connected

    def extract_split_merge_regions(self) -> List[Dict[str, Any]]:
        """Identify subgraph regions between split and merge points.

        A *split point* is a node with out-degree ≥ 2 and a *merge
        point* is a node with in-degree ≥ 2.  Each region is the set
        of nodes between a split point and its nearest downstream
        merge point.

        Returns
        -------
        List of dicts with keys ``"split"``, ``"merge"``, and ``"nodes"``.
        """
        regions: List[Dict[str, Any]] = []
        split_nodes = [
            nid
            for nid in self._dag.nodes
            if len(self._dag.successors(nid)) >= 2
        ]
        merge_nodes_set: Set[str] = {
            nid
            for nid in self._dag.nodes
            if len(self._dag.predecessors(nid)) >= 2
        }

        for split_id in split_nodes:
            reachable = ReachabilityAnalyzer(self._dag).reachable_from({split_id})
            # Find the nearest merge point in topological order
            for nid in self._dag.topological_order():
                if nid in reachable and nid in merge_nodes_set:
                    between = self.extract_between({split_id}, {nid})
                    regions.append({
                        "split": split_id,
                        "merge": nid,
                        "nodes": between,
                    })
                    break

        return regions

    def extract_induced_subgraph(self, node_ids: Set[str]) -> Set[str]:
        """Return *node_ids* filtered to those present in the DAG.

        The induced subgraph edges are implicitly determined by the
        DAG structure; this method validates that the requested nodes
        actually exist.
        """
        return {nid for nid in node_ids if self._dag.has_node(nid)}


# ===================================================================
#  LevelAssigner
# ===================================================================


@dataclass
class LevelAssignment:
    """Result of topological level assignment.

    Attributes
    ----------
    node_to_level
        Maps each ``node_id`` to its topological level (0-based).
    level_to_nodes
        Maps each level (int) to the list of node IDs at that level.
    max_level
        The highest level in the DAG.
    width
        Maximum number of nodes at any single level (anti-chain upper
        bound).
    """

    node_to_level: Dict[str, int] = field(default_factory=dict)
    level_to_nodes: Dict[int, List[str]] = field(
        default_factory=lambda: defaultdict(list),
    )
    max_level: int = 0
    width: int = 0


class LevelAssigner:
    """Assign topological levels to nodes in the PI-DAG.

    Level 0 is assigned to source nodes (no predecessors).  Every other
    node is placed at ``max(predecessor levels) + 1``, producing the
    standard *longest-path layering* used in Sugiyama-style graph layout.
    """

    def __init__(self, dag: "PIDAG") -> None:
        self._dag = dag

    def assign_levels(self) -> LevelAssignment:
        """Compute and return the level assignment for the entire DAG.

        Returns
        -------
        :class:`LevelAssignment` with level mappings and statistics.
        """
        topo = self._dag.topological_order()
        node_to_level: Dict[str, int] = {}

        for nid in topo:
            preds = self._dag.predecessors(nid)
            if not preds:
                node_to_level[nid] = 0
            else:
                node_to_level[nid] = (
                    max(node_to_level.get(p, 0) for p in preds) + 1
                )

        level_to_nodes: Dict[int, List[str]] = defaultdict(list)
        for nid, lvl in node_to_level.items():
            level_to_nodes[lvl].append(nid)

        max_level = max(node_to_level.values()) if node_to_level else 0
        width = (
            max(len(ns) for ns in level_to_nodes.values())
            if level_to_nodes
            else 0
        )

        return LevelAssignment(
            node_to_level=node_to_level,
            level_to_nodes=dict(level_to_nodes),
            max_level=max_level,
            width=width,
        )

    def nodes_at_level(self, level: int) -> List[str]:
        """Return node IDs at the given topological level."""
        return self.assign_levels().level_to_nodes.get(level, [])

    def depth(self) -> int:
        """Return the DAG depth (maximum topological level)."""
        return self.assign_levels().max_level


# ===================================================================
#  CriticalPathAnalyzer
# ===================================================================


@dataclass
class CriticalPathResult:
    """Result of critical-path analysis.

    Attributes
    ----------
    weight
        Total weight of the critical path.
    path
        Ordered list of node IDs on the critical path.
    node_slack
        Maps each node to its *slack* (latest start − earliest start).
        Nodes with zero slack lie on the critical path.
    """

    weight: float = 0.0
    path: List[str] = field(default_factory=list)
    node_slack: Dict[str, float] = field(default_factory=dict)


class CriticalPathAnalyzer:
    """Find the longest (critical) path through the PI-DAG.

    In the context of leakage auditing the critical path represents the
    pipeline path with the greatest potential for information leakage,
    typically quantified by edge capacity or provenance fraction.
    """

    def __init__(self, dag: "PIDAG") -> None:
        self._dag = dag

    def find_critical_path(
        self,
        weight_fn: Optional[Callable[[str, str], float]] = None,
    ) -> CriticalPathResult:
        """Compute the single critical (longest weighted) path.

        Parameters
        ----------
        weight_fn
            ``(source_id, target_id) -> weight`` for an edge.
            Defaults to ``1.0`` per edge (longest by hop count).

        Returns
        -------
        :class:`CriticalPathResult` with the path, total weight, and
        per-node slack values.
        """
        if weight_fn is None:
            weight_fn = lambda s, t: 1.0

        topo = self._dag.topological_order()
        if not topo:
            return CriticalPathResult()

        # Forward pass — compute earliest start time for each node
        earliest: Dict[str, float] = {nid: 0.0 for nid in topo}
        pred_on_path: Dict[str, Optional[str]] = {nid: None for nid in topo}

        for nid in topo:
            for succ in self._dag.successors(nid):
                w = weight_fn(nid, succ)
                candidate = earliest[nid] + w
                if candidate > earliest[succ]:
                    earliest[succ] = candidate
                    pred_on_path[succ] = nid

        # Identify end of the critical path
        end_node = max(topo, key=lambda n: earliest[n])
        total_weight = earliest[end_node]

        # Reconstruct path by following predecessors
        path: List[str] = []
        cur: Optional[str] = end_node
        while cur is not None:
            path.append(cur)
            cur = pred_on_path[cur]
        path.reverse()

        # Backward pass — compute latest start time for each node
        latest: Dict[str, float] = {nid: total_weight for nid in topo}
        for nid in reversed(topo):
            for pred_id in self._dag.predecessors(nid):
                w = weight_fn(pred_id, nid)
                candidate = latest[nid] - w
                if candidate < latest[pred_id]:
                    latest[pred_id] = candidate

        # Slack = latest − earliest; zero-slack nodes are critical
        node_slack: Dict[str, float] = {
            nid: latest[nid] - earliest[nid] for nid in topo
        }

        return CriticalPathResult(
            weight=total_weight,
            path=path,
            node_slack=node_slack,
        )

    def find_longest_path(self) -> List[str]:
        """Return the longest path by hop count (unweighted)."""
        return self.find_critical_path().path

    def node_criticality(self) -> Dict[str, float]:
        """Compute a criticality score for each node.

        Criticality is ``1.0 − normalised_slack``.  Nodes on the
        critical path have criticality ``1.0``; nodes with maximum
        slack have criticality ``0.0``.

        Returns
        -------
        Dict mapping ``node_id`` to a criticality score in ``[0, 1]``.
        """
        result = self.find_critical_path()
        if not result.node_slack:
            return {}
        max_slack = max(result.node_slack.values())
        if max_slack == 0.0:
            return {nid: 1.0 for nid in result.node_slack}
        return {
            nid: 1.0 - (slack / max_slack)
            for nid, slack in result.node_slack.items()
        }
