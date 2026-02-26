"""
D-separation based branch pruning for MCTS.

Precomputes a d-separation oracle from the causal DAG and provides
incremental updates as evidence (partial shock assignment) changes.
Implements the Bayes-ball algorithm for active trail detection and
tracks pruning history for analysis.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np

try:
    import networkx as nx
except ImportError:
    nx = None  # type: ignore


# -----------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------

@dataclass
class PruningRecord:
    """Record of a single pruning decision."""

    variable: str
    target: str
    evidence: FrozenSet[str]
    is_pruned: bool
    timestamp: float
    method: str = "bayes_ball"


@dataclass
class DSepOracleEntry:
    """Precomputed d-separation results for a variable pair."""

    source: str
    target: str
    # Maps frozenset of evidence to d-sep result
    cached_results: Dict[FrozenSet[str], bool] = field(default_factory=dict)


# -----------------------------------------------------------------------
# DSeparationPruner
# -----------------------------------------------------------------------

class DSeparationPruner:
    """
    D-separation based branch pruner for MCTS search.

    Given a causal DAG, this class provides efficient d-separation queries
    to determine which shock variables are irrelevant to the target loss
    given a partial assignment. Irrelevant branches are pruned to focus
    search on the effective action space.

    Parameters
    ----------
    dag : nx.DiGraph or None
        Causal DAG. Can also be set later via ``set_dag``.
    precompute : bool
        If True, precompute the d-separation oracle upon initialization.
    """

    def __init__(
        self,
        dag: Any = None,
        precompute: bool = False,
    ) -> None:
        self._dag: Any = dag
        self._oracle: Dict[Tuple[str, str], DSepOracleEntry] = {}
        self._history: List[PruningRecord] = []
        self._ancestors_cache: Dict[str, Set[str]] = {}
        self._descendants_cache: Dict[str, Set[str]] = {}
        self._moral_graph_cache: Optional[Any] = None
        self._topological_order: Optional[List[str]] = None

        if dag is not None and precompute:
            self.precompute_dsep_oracle(dag)

    # ------------------------------------------------------------------
    # DAG management
    # ------------------------------------------------------------------

    def set_dag(self, dag: Any) -> None:
        """Set or update the causal DAG and clear caches."""
        self._dag = dag
        self._oracle.clear()
        self._ancestors_cache.clear()
        self._descendants_cache.clear()
        self._moral_graph_cache = None
        self._topological_order = None

    # ------------------------------------------------------------------
    # Precomputation
    # ------------------------------------------------------------------

    def precompute_dsep_oracle(self, dag: Any) -> None:
        """
        Precompute d-separation relationships for all variable pairs.

        For each pair (u, v) in the DAG, computes whether u is d-separated
        from v given the empty evidence set. Further evidence sets are
        computed lazily and cached.

        Parameters
        ----------
        dag : nx.DiGraph
            The causal DAG.
        """
        self._dag = dag
        nodes = list(dag.nodes())

        # Cache ancestors and descendants
        for node in nodes:
            self._ancestors_cache[node] = set(nx.ancestors(dag, node))
            self._descendants_cache[node] = set(nx.descendants(dag, node))

        # Topological order
        self._topological_order = list(nx.topological_sort(dag))

        # Precompute moral graph (for alternative d-sep check)
        self._moral_graph_cache = self._build_moral_graph(dag)

        # Precompute empty-evidence d-separation
        for i, src in enumerate(nodes):
            for tgt in nodes[i + 1 :]:
                entry = DSepOracleEntry(source=src, target=tgt)
                empty_ev: FrozenSet[str] = frozenset()
                entry.cached_results[empty_ev] = self._bayes_ball_dsep(
                    src, tgt, set(), dag
                )
                self._oracle[(src, tgt)] = entry
                # Symmetric entry
                entry_rev = DSepOracleEntry(source=tgt, target=src)
                entry_rev.cached_results[empty_ev] = entry.cached_results[empty_ev]
                self._oracle[(tgt, src)] = entry_rev

    def _build_moral_graph(self, dag: Any) -> Any:
        """
        Build the moral graph from the DAG.

        The moral graph is formed by:
        1. For every node with multiple parents, connecting all parent pairs.
        2. Dropping edge directions.
        """
        if nx is None:
            return None

        moral = nx.Graph()
        moral.add_nodes_from(dag.nodes())

        for u, v in dag.edges():
            moral.add_edge(u, v)

        for node in dag.nodes():
            parents = list(dag.predecessors(node))
            for i in range(len(parents)):
                for j in range(i + 1, len(parents)):
                    moral.add_edge(parents[i], parents[j])

        return moral

    # ------------------------------------------------------------------
    # Core d-separation query
    # ------------------------------------------------------------------

    def is_prunable(
        self,
        variable: str,
        target: str,
        evidence: Dict[str, float],
        dag: Optional[Any] = None,
    ) -> bool:
        """
        Check if ``variable`` is d-separated from ``target`` given
        the evidence set, and therefore can be pruned.

        Parameters
        ----------
        variable : str
            The shock variable to check.
        target : str
            Target loss variable.
        evidence : dict
            Current partial assignment (variable -> value).
        dag : nx.DiGraph or None
            Causal DAG. Uses stored DAG if None.

        Returns
        -------
        bool
            True if the variable can be safely pruned.
        """
        if dag is None:
            dag = self._dag
        if dag is None:
            return False

        evidence_set = set(evidence.keys())
        evidence_frozen = frozenset(evidence_set)

        # Check oracle cache
        pair = (variable, target)
        if pair in self._oracle:
            if evidence_frozen in self._oracle[pair].cached_results:
                result = self._oracle[pair].cached_results[evidence_frozen]
                self._record_pruning(variable, target, evidence_frozen, result)
                return result

        # Compute d-separation
        result = self._bayes_ball_dsep(variable, target, evidence_set, dag)

        # Cache result
        if pair not in self._oracle:
            self._oracle[pair] = DSepOracleEntry(source=variable, target=target)
        self._oracle[pair].cached_results[evidence_frozen] = result

        self._record_pruning(variable, target, evidence_frozen, result)
        return result

    # ------------------------------------------------------------------
    # Get relevant variables
    # ------------------------------------------------------------------

    def get_relevant_variables(
        self,
        target: str,
        evidence: Dict[str, float],
        dag: Optional[Any] = None,
    ) -> List[str]:
        """
        Return the list of variables NOT d-separated from target.

        These are the variables that may influence the target given the
        current evidence and should not be pruned.

        Parameters
        ----------
        target : str
            Target variable.
        evidence : dict
            Current partial assignment.
        dag : nx.DiGraph or None
            Causal DAG.

        Returns
        -------
        list of str
            Relevant (non-prunable) variables.
        """
        if dag is None:
            dag = self._dag
        if dag is None:
            return []

        relevant = []
        evidence_set = set(evidence.keys())

        for node in dag.nodes():
            if node == target or node in evidence:
                continue
            if not self._bayes_ball_dsep(node, target, evidence_set, dag):
                relevant.append(node)

        return relevant

    # ------------------------------------------------------------------
    # Active trail detection (Bayes-ball)
    # ------------------------------------------------------------------

    def get_active_trails(
        self,
        source: str,
        evidence: Dict[str, float],
        dag: Optional[Any] = None,
    ) -> Dict[str, List[List[str]]]:
        """
        Find all active trails from source to every reachable node.

        Returns a mapping from reachable node to a list of trails.
        Each trail is a list of variable names.

        Parameters
        ----------
        source : str
            Starting variable.
        evidence : dict
            Current evidence/observed variables.
        dag : nx.DiGraph or None
            Causal DAG.

        Returns
        -------
        dict
            Maps reachable node -> list of trails (each trail is a list of node names).
        """
        if dag is None:
            dag = self._dag
        if dag is None:
            return {}

        evidence_set = set(evidence.keys())

        desc_of_evidence = set()
        for ev in evidence_set:
            if ev in dag:
                desc_of_evidence.update(nx.descendants(dag, ev))
                desc_of_evidence.add(ev)

        # BFS with path tracking
        # State: (current_node, direction, path)
        trails: Dict[str, List[List[str]]] = defaultdict(list)
        visited: Set[Tuple[str, str]] = set()
        queue: deque = deque()

        queue.append((source, "up", [source]))
        queue.append((source, "down", [source]))

        while queue:
            node, direction, path = queue.popleft()

            state_key = (node, direction)
            if state_key in visited:
                continue
            visited.add(state_key)

            if node != source:
                trails[node].append(list(path))

            if direction == "up" and node not in evidence_set:
                for parent in dag.predecessors(node):
                    new_path = path + [parent]
                    queue.append((parent, "up", new_path))
                for child in dag.successors(node):
                    new_path = path + [child]
                    queue.append((child, "down", new_path))

            elif direction == "down":
                if node not in evidence_set:
                    for child in dag.successors(node):
                        new_path = path + [child]
                        queue.append((child, "down", new_path))
                if node in desc_of_evidence:
                    for parent in dag.predecessors(node):
                        new_path = path + [parent]
                        queue.append((parent, "up", new_path))

        return dict(trails)

    # ------------------------------------------------------------------
    # Bayes-ball d-separation implementation
    # ------------------------------------------------------------------

    def _bayes_ball_dsep(
        self,
        source: str,
        target: str,
        evidence: Set[str],
        dag: Any,
    ) -> bool:
        """
        Determine d-separation using the Bayes-ball algorithm.

        source ⊥ target | evidence if and only if the Bayes-ball
        started at source cannot reach target.

        Parameters
        ----------
        source : str
            Source variable.
        target : str
            Target variable.
        evidence : set of str
            Evidence / conditioning variables.
        dag : nx.DiGraph
            Causal DAG.

        Returns
        -------
        bool
            True if source and target are d-separated given evidence.
        """
        if source not in dag or target not in dag:
            return True
        if source == target:
            return False

        # Precompute evidence descendants
        desc_of_evidence: Set[str] = set()
        for ev in evidence:
            if ev in dag:
                desc_of_evidence.update(nx.descendants(dag, ev))
                desc_of_evidence.add(ev)

        visited_from_child: Set[str] = set()
        visited_from_parent: Set[str] = set()

        # Queue entries: (node, came_from_child: bool)
        queue: deque = deque()
        queue.append((source, True))   # as if a child sent the ball up
        queue.append((source, False))  # as if a parent sent the ball down

        while queue:
            node, from_child = queue.popleft()

            if node == target:
                return False  # ball reached target → not d-separated

            if from_child:
                if node in visited_from_child:
                    continue
                visited_from_child.add(node)

                if node not in evidence:
                    # Ball passes through: send to parents (up) and children (down)
                    for parent in dag.predecessors(node):
                        queue.append((parent, True))
                    for child in dag.successors(node):
                        queue.append((child, False))
                # If observed: ball is blocked on chain/fork, but
                # need to check if descendant of evidence (for collider activation)
                # Actually, if node IS in evidence, ball bounces back to parents
                # only if it arrived from a child (v-structure activation)
                # — this is handled by from_child=True + node in evidence below

            else:
                # Ball traveling down (from parent)
                if node in visited_from_parent:
                    continue
                visited_from_parent.add(node)

                if node not in evidence:
                    # Not observed: pass ball to children
                    for child in dag.successors(node):
                        queue.append((child, False))
                # If observed or descendant of observed: activate v-structure
                if node in desc_of_evidence:
                    for parent in dag.predecessors(node):
                        queue.append((parent, True))

        return True  # ball never reached target → d-separated

    # ------------------------------------------------------------------
    # Pruning ratio computation
    # ------------------------------------------------------------------

    def compute_pruning_ratio(
        self,
        dag: Any,
        target: str,
        evidence: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Compute the fraction of variables that can be pruned.

        Parameters
        ----------
        dag : nx.DiGraph
            Causal DAG.
        target : str
            Target variable.
        evidence : dict or None
            Current evidence. Empty dict if None.

        Returns
        -------
        dict
            Contains 'total_vars', 'prunable', 'relevant',
            'pruning_ratio', and lists 'prunable_vars', 'relevant_vars'.
        """
        if evidence is None:
            evidence = {}

        evidence_set = set(evidence.keys())
        all_vars = [n for n in dag.nodes() if n != target and n not in evidence]

        prunable_vars = []
        relevant_vars = []

        for var in all_vars:
            if self._bayes_ball_dsep(var, target, evidence_set, dag):
                prunable_vars.append(var)
            else:
                relevant_vars.append(var)

        total = len(all_vars)
        ratio = len(prunable_vars) / total if total > 0 else 0.0

        return {
            "total_vars": total,
            "prunable": len(prunable_vars),
            "relevant": len(relevant_vars),
            "pruning_ratio": ratio,
            "prunable_vars": prunable_vars,
            "relevant_vars": relevant_vars,
        }

    # ------------------------------------------------------------------
    # Incremental d-sep update
    # ------------------------------------------------------------------

    def incremental_update(
        self,
        new_evidence_var: str,
        target: str,
        previous_evidence: Dict[str, float],
        dag: Optional[Any] = None,
    ) -> Dict[str, bool]:
        """
        Incrementally update d-separation results when a new variable
        is added to the evidence.

        Instead of recomputing d-sep for all variables from scratch,
        only re-evaluate variables that might change status due to the
        new evidence.

        Parameters
        ----------
        new_evidence_var : str
            Variable newly added to the evidence set.
        target : str
            Target variable.
        previous_evidence : dict
            Evidence before the update.
        dag : nx.DiGraph or None
            Causal DAG.

        Returns
        -------
        dict
            Maps variable name to updated d-separation status (True = prunable).
        """
        if dag is None:
            dag = self._dag
        if dag is None:
            return {}

        updated_evidence = dict(previous_evidence)
        updated_evidence[new_evidence_var] = 0.0  # value doesn't matter for d-sep

        evidence_set = set(updated_evidence.keys())

        # Identify candidates that might change status:
        # 1. Neighbors of the new evidence variable in the moral graph
        # 2. Ancestors and descendants of the new evidence variable
        candidates: Set[str] = set()

        if new_evidence_var in dag:
            # Direct neighbors
            candidates.update(dag.predecessors(new_evidence_var))
            candidates.update(dag.successors(new_evidence_var))

            # Ancestors
            if new_evidence_var in self._ancestors_cache:
                candidates.update(self._ancestors_cache[new_evidence_var])
            else:
                candidates.update(nx.ancestors(dag, new_evidence_var))

            # Descendants
            if new_evidence_var in self._descendants_cache:
                candidates.update(self._descendants_cache[new_evidence_var])
            else:
                candidates.update(nx.descendants(dag, new_evidence_var))

        # Also include all variables adjacent in the moral graph
        if self._moral_graph_cache is not None and new_evidence_var in self._moral_graph_cache:
            candidates.update(self._moral_graph_cache.neighbors(new_evidence_var))

        candidates.discard(target)
        candidates -= evidence_set

        results: Dict[str, bool] = {}
        evidence_frozen = frozenset(evidence_set)

        for var in candidates:
            if var not in dag:
                continue
            is_dsep = self._bayes_ball_dsep(var, target, evidence_set, dag)
            results[var] = is_dsep

            # Update oracle cache
            pair = (var, target)
            if pair not in self._oracle:
                self._oracle[pair] = DSepOracleEntry(source=var, target=target)
            self._oracle[pair].cached_results[evidence_frozen] = is_dsep

        return results

    # ------------------------------------------------------------------
    # Markov blanket
    # ------------------------------------------------------------------

    def get_markov_blanket(
        self,
        variable: str,
        dag: Optional[Any] = None,
    ) -> Set[str]:
        """
        Compute the Markov blanket of a variable.

        The Markov blanket consists of parents, children, and
        co-parents (other parents of the variable's children).

        Parameters
        ----------
        variable : str
            The variable whose Markov blanket to compute.
        dag : nx.DiGraph or None
            Causal DAG.

        Returns
        -------
        set of str
            Markov blanket nodes.
        """
        if dag is None:
            dag = self._dag
        if dag is None:
            return set()

        if variable not in dag:
            return set()

        blanket: Set[str] = set()

        # Parents
        blanket.update(dag.predecessors(variable))

        # Children
        children = set(dag.successors(variable))
        blanket.update(children)

        # Co-parents (other parents of children)
        for child in children:
            blanket.update(dag.predecessors(child))

        blanket.discard(variable)
        return blanket

    def get_minimal_separating_set(
        self,
        source: str,
        target: str,
        dag: Optional[Any] = None,
    ) -> Optional[Set[str]]:
        """
        Find a minimal set of variables that d-separates source from target.

        Uses an iterative approach: start with the Markov blanket of source
        and remove variables one at a time if d-separation is maintained.

        Parameters
        ----------
        source : str
            Source variable.
        target : str
            Target variable.
        dag : nx.DiGraph or None
            Causal DAG.

        Returns
        -------
        set of str or None
            Minimal separating set, or None if no finite set suffices.
        """
        if dag is None:
            dag = self._dag
        if dag is None:
            return None

        if source not in dag or target not in dag:
            return None

        # Start with all non-source, non-target nodes
        all_nodes = set(dag.nodes()) - {source, target}

        # First check if d-separation is achievable at all
        if not self._bayes_ball_dsep(source, target, all_nodes, dag):
            return None

        # Greedy removal
        sep_set = set(all_nodes)
        for node in list(all_nodes):
            candidate = sep_set - {node}
            if self._bayes_ball_dsep(source, target, candidate, dag):
                sep_set = candidate

        return sep_set

    # ------------------------------------------------------------------
    # History and diagnostics
    # ------------------------------------------------------------------

    def _record_pruning(
        self,
        variable: str,
        target: str,
        evidence: FrozenSet[str],
        is_pruned: bool,
    ) -> None:
        """Record a pruning decision in the history."""
        self._history.append(
            PruningRecord(
                variable=variable,
                target=target,
                evidence=evidence,
                is_pruned=is_pruned,
                timestamp=time.time(),
            )
        )

    def get_pruning_history(
        self, last_n: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Return the pruning history as a list of dictionaries.

        Parameters
        ----------
        last_n : int or None
            Return only the last n records.
        """
        records = self._history if last_n is None else self._history[-last_n:]
        return [
            {
                "variable": r.variable,
                "target": r.target,
                "evidence": sorted(r.evidence),
                "is_pruned": r.is_pruned,
                "timestamp": r.timestamp,
                "method": r.method,
            }
            for r in records
        ]

    def get_pruning_summary(self) -> Dict[str, Any]:
        """
        Return aggregate pruning statistics.

        Returns
        -------
        dict
            Contains total queries, prune count, prune rate per variable.
        """
        total = len(self._history)
        pruned = sum(1 for r in self._history if r.is_pruned)

        per_variable: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"queries": 0, "pruned": 0}
        )
        for r in self._history:
            per_variable[r.variable]["queries"] += 1
            if r.is_pruned:
                per_variable[r.variable]["pruned"] += 1

        return {
            "total_queries": total,
            "total_pruned": pruned,
            "prune_rate": pruned / total if total > 0 else 0.0,
            "per_variable": dict(per_variable),
            "oracle_cache_size": sum(
                len(e.cached_results) for e in self._oracle.values()
            ),
        }

    def clear_history(self) -> None:
        """Clear the pruning history."""
        self._history.clear()

    def clear_cache(self) -> None:
        """Clear oracle caches."""
        self._oracle.clear()
        self._ancestors_cache.clear()
        self._descendants_cache.clear()
        self._moral_graph_cache = None
