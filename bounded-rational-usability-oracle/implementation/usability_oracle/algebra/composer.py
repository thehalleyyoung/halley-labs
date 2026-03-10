"""
usability_oracle.algebra.composer — Task-graph-level cost composition.

Given a task dependency graph (as a :class:`networkx.DiGraph`) and a mapping
from node IDs to :class:`CostElement` instances, the :class:`TaskGraphComposer`
recursively applies ⊕ (sequential) and ⊗ (parallel) operators to compute
the total cost of the task.

Algorithm
---------
1. **Topological sort** the DAG.
2. **Detect parallel groups** — nodes at the same topological level that
   share no dependency edge form a parallel group.
3. **Compose parallel groups** using ⊗.
4. **Compose levels sequentially** using ⊕.

The result is a single :class:`CostElement` representing the total cost
of the entire task graph.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

from usability_oracle.algebra.models import CostElement
from usability_oracle.algebra.sequential import SequentialComposer
from usability_oracle.algebra.parallel import ParallelComposer


# ---------------------------------------------------------------------------
# TaskGraphComposer
# ---------------------------------------------------------------------------


class TaskGraphComposer:
    """Compose a task dependency graph into a single :class:`CostElement`.

    The graph is a :class:`networkx.DiGraph` where each node has a
    corresponding :class:`CostElement` in the ``cost_map``.

    Usage::

        G = nx.DiGraph()
        G.add_edges_from([("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")])
        costs = {
            "A": CostElement(1.0, 0.1, 0.0, 0.01),
            "B": CostElement(2.0, 0.2, 0.1, 0.02),
            "C": CostElement(1.5, 0.15, 0.0, 0.01),
            "D": CostElement(0.5, 0.05, 0.0, 0.01),
        }
        composer = TaskGraphComposer()
        total = composer.compose(G, costs)
    """

    def __init__(
        self,
        *,
        default_coupling: float = 0.0,
        default_interference: float = 0.0,
    ) -> None:
        """
        Parameters
        ----------
        default_coupling : float
            Default sequential coupling parameter for ⊕.
        default_interference : float
            Default parallel interference parameter for ⊗.
        """
        self._coupling = default_coupling
        self._interference = default_interference
        self._seq = SequentialComposer()
        self._par = ParallelComposer()

    # -- main entry point ----------------------------------------------------

    def compose(
        self,
        task_graph: nx.DiGraph,
        cost_map: Dict[str, CostElement],
    ) -> CostElement:
        """Compose the entire task graph into a single :class:`CostElement`.

        Parameters
        ----------
        task_graph : nx.DiGraph
            Directed acyclic graph of task dependencies.
        cost_map : dict[str, CostElement]
            Mapping from node IDs to their individual costs.

        Returns
        -------
        CostElement
            Total composed cost of the task graph.

        Raises
        ------
        ValueError
            If the graph contains a cycle or nodes are missing from
            ``cost_map``.
        """
        if not task_graph.nodes:
            return CostElement.zero()

        if not nx.is_directed_acyclic_graph(task_graph):
            raise ValueError("Task graph contains a cycle.")

        # Validate cost_map coverage
        missing = set(task_graph.nodes) - set(cost_map.keys())
        if missing:
            raise ValueError(
                f"Nodes missing from cost_map: {missing}"
            )

        return self._topological_compose(task_graph, cost_map)

    # -- topological composition ---------------------------------------------

    def _topological_compose(
        self,
        graph: nx.DiGraph,
        costs: Dict[str, CostElement],
    ) -> CostElement:
        """Compose by topological levels.

        Nodes at the same level with no mutual dependencies are composed
        in parallel; levels are composed sequentially.
        """
        parallel_groups = self._detect_parallel_groups(graph)

        if not parallel_groups:
            return CostElement.zero()

        level_costs: List[CostElement] = []
        for group in parallel_groups:
            group_cost = self._compose_group(group, costs)
            level_costs.append(group_cost)

        return self._compose_path(level_costs)

    # -- parallel group detection --------------------------------------------

    def _detect_parallel_groups(
        self, graph: nx.DiGraph
    ) -> List[Set[str]]:
        """Partition nodes into topological levels (parallel groups).

        Two nodes are in the same group if they have the same topological
        depth (longest path from any root).

        Parameters
        ----------
        graph : nx.DiGraph

        Returns
        -------
        list[set[str]]
            Groups ordered by increasing topological depth.
        """
        topo_order = list(nx.topological_sort(graph))
        depth: Dict[str, int] = {}

        for node in topo_order:
            preds = list(graph.predecessors(node))
            if not preds:
                depth[node] = 0
            else:
                depth[node] = max(depth[p] for p in preds) + 1

        # Group by depth
        groups_map: Dict[int, Set[str]] = defaultdict(set)
        for node, d in depth.items():
            groups_map[d].add(node)

        return [groups_map[d] for d in sorted(groups_map)]

    # -- group composition ---------------------------------------------------

    def _compose_group(
        self,
        group: Set[str],
        costs: Dict[str, CostElement],
    ) -> CostElement:
        """Compose a parallel group (all nodes at the same level).

        Parameters
        ----------
        group : set[str]
            Node IDs that execute concurrently.
        costs : dict[str, CostElement]
            Cost map.

        Returns
        -------
        CostElement
        """
        if not group:
            return CostElement.zero()

        elements = [costs[nid] for nid in sorted(group)]

        if len(elements) == 1:
            return elements[0]

        return self._par.compose_group(elements, interference=self._interference)

    # -- path composition ----------------------------------------------------

    def _compose_path(
        self,
        elements: List[CostElement],
    ) -> CostElement:
        """Sequentially compose a list of cost elements (a path).

        Parameters
        ----------
        elements : list[CostElement]
            Ordered cost elements to compose sequentially.

        Returns
        -------
        CostElement
        """
        return self._seq.compose_chain(
            elements,
            couplings=[self._coupling] * max(0, len(elements) - 1),
        )

    # -- advanced: weighted critical path ------------------------------------

    def critical_path_cost(
        self,
        task_graph: nx.DiGraph,
        cost_map: Dict[str, CostElement],
    ) -> Tuple[List[str], CostElement]:
        """Compute the critical path and its total cost.

        The critical path is the longest-mean-cost path through the DAG.

        Parameters
        ----------
        task_graph : nx.DiGraph
        cost_map : dict[str, CostElement]

        Returns
        -------
        (path, cost) : tuple[list[str], CostElement]
        """
        if not task_graph.nodes:
            return [], CostElement.zero()

        topo_order = list(nx.topological_sort(task_graph))

        # Longest-path DP
        dist: Dict[str, float] = {n: 0.0 for n in task_graph.nodes}
        pred: Dict[str, Optional[str]] = {n: None for n in task_graph.nodes}

        for node in topo_order:
            node_cost = cost_map[node].mu
            for succ in task_graph.successors(node):
                new_dist = dist[node] + node_cost
                if new_dist > dist[succ]:
                    dist[succ] = new_dist
                    pred[succ] = node

        # Find the endpoint with maximum distance + its own cost
        end_dists = {n: dist[n] + cost_map[n].mu for n in task_graph.nodes}
        end_node = max(end_dists, key=lambda k: end_dists[k])

        # Backtrack
        path: List[str] = []
        cur: Optional[str] = end_node
        while cur is not None:
            path.append(cur)
            cur = pred[cur]
        path.reverse()

        # Compose the path cost
        path_elements = [cost_map[n] for n in path]
        total_cost = self._seq.compose_chain(
            path_elements,
            couplings=[self._coupling] * max(0, len(path_elements) - 1),
        )

        return path, total_cost

    # -- graph analytics -----------------------------------------------------

    def parallelism_factor(self, task_graph: nx.DiGraph) -> float:
        """Estimate the degree of parallelism in the task graph.

        Computed as ``total_nodes / critical_path_length``.  A value of 1
        means the graph is fully serial; higher values indicate more
        parallelism.

        Returns
        -------
        float
        """
        if not task_graph.nodes:
            return 1.0

        crit_path_len = nx.dag_longest_path_length(task_graph) + 1
        return len(task_graph.nodes) / crit_path_len

    def bottleneck_nodes(
        self,
        task_graph: nx.DiGraph,
        cost_map: Dict[str, CostElement],
        top_k: int = 3,
    ) -> List[Tuple[str, float]]:
        """Identify the top-*k* bottleneck nodes by cost contribution.

        A node's bottleneck score is the product of its own cost and the
        number of nodes that depend on it (directly or transitively).

        Parameters
        ----------
        task_graph : nx.DiGraph
        cost_map : dict[str, CostElement]
        top_k : int

        Returns
        -------
        list[tuple[str, float]]
            (node_id, bottleneck_score) pairs, sorted descending.
        """
        scores: List[Tuple[str, float]] = []
        for node in task_graph.nodes:
            # Number of descendants (transitive dependents)
            n_desc = len(nx.descendants(task_graph, node))
            node_cost = cost_map[node].mu
            score = node_cost * (1 + n_desc)
            scores.append((node, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def subgraph_cost(
        self,
        task_graph: nx.DiGraph,
        cost_map: Dict[str, CostElement],
        nodes: Set[str],
    ) -> CostElement:
        """Compute the cost of a subgraph induced by *nodes*.

        Parameters
        ----------
        task_graph : nx.DiGraph
        cost_map : dict[str, CostElement]
        nodes : set[str]

        Returns
        -------
        CostElement
        """
        subgraph = task_graph.subgraph(nodes).copy()
        sub_costs = {n: cost_map[n] for n in nodes}
        return self.compose(subgraph, sub_costs)
