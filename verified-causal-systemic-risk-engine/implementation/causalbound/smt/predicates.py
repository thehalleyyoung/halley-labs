"""
GraphPredicateEncoder: encode graph-theoretic predicates as SMT formulas.

Translates d-separation, ancestry, path existence, Markov blanket
membership, intervention targets, and topological ordering into
Z3 Boolean / QF_LRA constraints suitable for incremental verification.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, FrozenSet, Iterable, List, Optional, Set, Tuple

import z3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _node_pair(a: str, b: str) -> str:
    return f"{a}__{b}"


def _var_name(prefix: str, *parts: str) -> str:
    sanitised = [p.replace(" ", "_").replace(".", "_") for p in parts]
    return f"{prefix}_{'_'.join(sanitised)}"


# ---------------------------------------------------------------------------
# Graph data structure used throughout
# ---------------------------------------------------------------------------

class DAG:
    """
    Lightweight directed acyclic graph representation.

    Stores adjacency lists for both forward (children) and backward
    (parent) directions, plus caches for ancestor / descendant sets.
    """

    def __init__(self, edges: Iterable[Tuple[str, str]]) -> None:
        self.nodes: Set[str] = set()
        self.edge_set: Set[Tuple[str, str]] = set()
        self._children: Dict[str, Set[str]] = defaultdict(set)
        self._parents: Dict[str, Set[str]] = defaultdict(set)

        for u, v in edges:
            self.nodes.add(u)
            self.nodes.add(v)
            self.edge_set.add((u, v))
            self._children[u].add(v)
            self._parents[v].add(u)

    def children(self, node: str) -> Set[str]:
        return self._children.get(node, set())

    def parents(self, node: str) -> Set[str]:
        return self._parents.get(node, set())

    def ancestors(self, node: str) -> Set[str]:
        visited: Set[str] = set()
        queue = deque(self.parents(node))
        while queue:
            cur = queue.popleft()
            if cur not in visited:
                visited.add(cur)
                queue.extend(self.parents(cur))
        return visited

    def descendants(self, node: str) -> Set[str]:
        visited: Set[str] = set()
        queue = deque(self.children(node))
        while queue:
            cur = queue.popleft()
            if cur not in visited:
                visited.add(cur)
                queue.extend(self.children(cur))
        return visited

    def has_edge(self, u: str, v: str) -> bool:
        return (u, v) in self.edge_set

    def topological_sort(self) -> List[str]:
        in_degree: Dict[str, int] = {n: 0 for n in self.nodes}
        for _, v in self.edge_set:
            in_degree[v] += 1
        queue = deque(n for n in sorted(self.nodes) if in_degree[n] == 0)
        order: List[str] = []
        while queue:
            cur = queue.popleft()
            order.append(cur)
            for ch in sorted(self.children(cur)):
                in_degree[ch] -= 1
                if in_degree[ch] == 0:
                    queue.append(ch)
        return order

    def markov_blanket(self, node: str) -> Set[str]:
        """Parents ∪ children ∪ co-parents of children."""
        mb: Set[str] = set()
        mb.update(self.parents(node))
        for ch in self.children(node):
            mb.add(ch)
            mb.update(self.parents(ch))
        mb.discard(node)
        return mb


# ---------------------------------------------------------------------------
# GraphPredicateEncoder
# ---------------------------------------------------------------------------

class GraphPredicateEncoder:
    """
    Encode graph-theoretic predicates as Z3 formulas.

    Each ``encode_*`` method returns a ``z3.BoolRef`` that, when added
    to the solver, asserts the corresponding graph property. Variables
    are cached so that the same symbolic variable is reused across
    different predicate calls within one encoder instance.

    Parameters
    ----------
    prefix : str
        Global variable-name prefix to prevent collisions when
        multiple encoders coexist.
    """

    def __init__(self, prefix: str = "gp") -> None:
        self._prefix = prefix
        self._var_cache: Dict[str, z3.ExprRef] = {}

    # ------------------------------------------------------------------
    # Variable helpers
    # ------------------------------------------------------------------

    def _bool(self, name: str) -> z3.BoolRef:
        key = f"{self._prefix}_{name}"
        if key not in self._var_cache:
            self._var_cache[key] = z3.Bool(key)
        return self._var_cache[key]  # type: ignore[return-value]

    def _int(self, name: str) -> z3.ArithRef:
        key = f"{self._prefix}_{name}"
        if key not in self._var_cache:
            self._var_cache[key] = z3.Int(key)
        return self._var_cache[key]  # type: ignore[return-value]

    def _real(self, name: str) -> z3.ArithRef:
        key = f"{self._prefix}_{name}"
        if key not in self._var_cache:
            self._var_cache[key] = z3.Real(key)
        return self._var_cache[key]  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # 1. d-separation   dsep(X, Y | Z)
    # ------------------------------------------------------------------

    def encode_dsep(
        self,
        x: str,
        y: str,
        z_set: Iterable[str],
        dag: Optional[DAG] = None,
        edges: Optional[Iterable[Tuple[str, str]]] = None,
    ) -> z3.BoolRef:
        """
        Encode dsep(X, Y | Z) as a reachability / blocking formula.

        Creates per-node Boolean ``reach`` and per-node direction
        indicators (``up``/``down``) to faithfully model the Bayes-Ball
        algorithm as SMT constraints.

        Parameters
        ----------
        x, y : str   Source and target nodes.
        z_set : iterable of str   Conditioning set.
        dag : DAG, optional   Pre-built DAG (takes priority).
        edges : iterable of (str, str), optional
            Edge list if *dag* is not provided.
        """
        if dag is None:
            if edges is None:
                raise ValueError("Provide either dag or edges")
            dag = DAG(edges)

        z_nodes = set(z_set)
        nodes = sorted(dag.nodes | {x, y} | z_nodes)
        tag = _var_name("dsep", x, y, *sorted(z_nodes))

        # Per-node reachability via "up" and "down" directions
        reach_up: Dict[str, z3.BoolRef] = {}
        reach_down: Dict[str, z3.BoolRef] = {}
        for n in nodes:
            reach_up[n] = self._bool(f"{tag}_up_{n}")
            reach_down[n] = self._bool(f"{tag}_dn_{n}")

        clauses: List[z3.BoolRef] = []

        # Source: x is reachable going "up" (toward ancestors)
        clauses.append(reach_up[x])
        clauses.append(reach_down[x])

        # Blocking rules (Bayes-Ball)
        for n in nodes:
            if n == x:
                continue

            in_z = n in z_nodes
            parent_reach = []
            child_reach = []

            # parents can reach n going "down"
            for p in dag.parents(n):
                if p in reach_down:
                    parent_reach.append(reach_down[p])
            # children can reach n going "up"
            for c in dag.children(n):
                if c in reach_up:
                    child_reach.append(reach_up[c])

            if in_z:
                # Observed node: blocks "down" pass-through
                clauses.append(z3.Not(reach_down[n]))
                # But "up" through observed node IS possible (explaining away)
                if parent_reach:
                    clauses.append(
                        reach_up[n] == z3.Or(*parent_reach)
                    )
                else:
                    clauses.append(z3.Not(reach_up[n]))
            else:
                # Unobserved node: passes through in both directions
                sources: List[z3.BoolRef] = []
                if parent_reach:
                    sources.extend(parent_reach)
                if child_reach:
                    sources.extend(child_reach)

                if sources:
                    clauses.append(reach_down[n] == z3.Or(*sources))
                    clauses.append(reach_up[n] == z3.Or(*sources))
                else:
                    clauses.append(z3.Not(reach_down[n]))
                    clauses.append(z3.Not(reach_up[n]))

        # d-separation ↔ y not reachable in either direction
        clauses.append(z3.Not(reach_up[y]))
        clauses.append(z3.Not(reach_down[y]))

        return z3.And(*clauses)

    def encode_dsep_negation(
        self,
        x: str,
        y: str,
        z_set: Iterable[str],
        dag: Optional[DAG] = None,
        edges: Optional[Iterable[Tuple[str, str]]] = None,
    ) -> z3.BoolRef:
        """Encode that X and Y are NOT d-separated given Z."""
        return z3.Not(self.encode_dsep(x, y, z_set, dag=dag, edges=edges))

    # ------------------------------------------------------------------
    # 2. Ancestral relation
    # ------------------------------------------------------------------

    def encode_ancestor(
        self,
        x: str,
        y: str,
        dag: Optional[DAG] = None,
        edges: Optional[Iterable[Tuple[str, str]]] = None,
    ) -> z3.BoolRef:
        """
        Encode that X is an ancestor of Y (there exists a directed
        path X → … → Y).

        Uses per-node reachability indicators with edge propagation.
        """
        if dag is None:
            if edges is None:
                raise ValueError("Provide either dag or edges")
            dag = DAG(edges)

        nodes = sorted(dag.nodes | {x, y})
        tag = _var_name("anc", x, y)

        reach: Dict[str, z3.BoolRef] = {}
        for n in nodes:
            reach[n] = self._bool(f"{tag}_r_{n}")

        clauses: List[z3.BoolRef] = []

        # x reaches itself
        clauses.append(reach[x])

        # Non-source nodes unreachable unless a parent is reachable
        for n in nodes:
            if n == x:
                continue
            parent_reachable = [reach[p] for p in dag.parents(n) if p in reach]
            if parent_reachable:
                clauses.append(
                    z3.Implies(z3.Or(*parent_reachable), reach[n])
                )
            # If no parents have reach vars it stays unconstrained but
            # the final claim gates the result.

        # Ancestor claim: y is reachable
        clauses.append(reach[y])

        return z3.And(*clauses)

    def encode_non_ancestor(
        self,
        x: str,
        y: str,
        dag: Optional[DAG] = None,
        edges: Optional[Iterable[Tuple[str, str]]] = None,
    ) -> z3.BoolRef:
        """Encode that X is NOT an ancestor of Y."""
        if dag is None:
            if edges is None:
                raise ValueError("Provide either dag or edges")
            dag = DAG(edges)

        nodes = sorted(dag.nodes | {x, y})
        tag = _var_name("nanc", x, y)

        reach: Dict[str, z3.BoolRef] = {}
        for n in nodes:
            reach[n] = self._bool(f"{tag}_r_{n}")

        clauses: List[z3.BoolRef] = []
        clauses.append(reach[x])

        for n in nodes:
            if n == x:
                continue
            parent_reachable = [reach[p] for p in dag.parents(n) if p in reach]
            if parent_reachable:
                clauses.append(
                    z3.Implies(z3.Or(*parent_reachable), reach[n])
                )

        # Claim: y NOT reachable
        clauses.append(z3.Not(reach[y]))

        return z3.And(*clauses)

    # ------------------------------------------------------------------
    # 3. Path existence (undirected)
    # ------------------------------------------------------------------

    def encode_path(
        self,
        x: str,
        y: str,
        dag: Optional[DAG] = None,
        edges: Optional[Iterable[Tuple[str, str]]] = None,
    ) -> z3.BoolRef:
        """
        Encode that there exists an undirected path between X and Y
        (ignoring edge direction).
        """
        if dag is None:
            if edges is None:
                raise ValueError("Provide either dag or edges")
            dag = DAG(edges)

        nodes = sorted(dag.nodes | {x, y})
        tag = _var_name("path", x, y)

        reach: Dict[str, z3.BoolRef] = {}
        for n in nodes:
            reach[n] = self._bool(f"{tag}_r_{n}")

        clauses: List[z3.BoolRef] = []
        clauses.append(reach[x])

        # Build undirected adjacency
        adj: Dict[str, Set[str]] = defaultdict(set)
        for u, v in dag.edge_set:
            adj[u].add(v)
            adj[v].add(u)

        for n in nodes:
            if n == x:
                continue
            neighbours = [reach[nb] for nb in adj.get(n, set()) if nb in reach]
            if neighbours:
                clauses.append(
                    z3.Implies(z3.Or(*neighbours), reach[n])
                )

        clauses.append(reach[y])
        return z3.And(*clauses)

    def encode_no_path(
        self,
        x: str,
        y: str,
        dag: Optional[DAG] = None,
        edges: Optional[Iterable[Tuple[str, str]]] = None,
    ) -> z3.BoolRef:
        """Encode that no undirected path exists between X and Y."""
        if dag is None:
            if edges is None:
                raise ValueError("Provide either dag or edges")
            dag = DAG(edges)

        nodes = sorted(dag.nodes | {x, y})
        tag = _var_name("nopath", x, y)

        reach: Dict[str, z3.BoolRef] = {}
        for n in nodes:
            reach[n] = self._bool(f"{tag}_r_{n}")

        clauses: List[z3.BoolRef] = []
        clauses.append(reach[x])

        adj: Dict[str, Set[str]] = defaultdict(set)
        for u, v in dag.edge_set:
            adj[u].add(v)
            adj[v].add(u)

        for n in nodes:
            if n == x:
                continue
            neighbours = [reach[nb] for nb in adj.get(n, set()) if nb in reach]
            if neighbours:
                clauses.append(
                    z3.Implies(z3.Or(*neighbours), reach[n])
                )

        clauses.append(z3.Not(reach[y]))
        return z3.And(*clauses)

    # ------------------------------------------------------------------
    # 4. Markov blanket membership
    # ------------------------------------------------------------------

    def encode_markov_blanket(
        self,
        x: str,
        dag: Optional[DAG] = None,
        edges: Optional[Iterable[Tuple[str, str]]] = None,
    ) -> z3.BoolRef:
        """
        Encode Markov blanket membership for node X.

        Creates Boolean indicators for each node: ``in_mb_<n>`` is true
        iff *n* belongs to MB(X).  MB(X) = parents(X) ∪ children(X)
        ∪ {parents of children of X}.
        """
        if dag is None:
            if edges is None:
                raise ValueError("Provide either dag or edges")
            dag = DAG(edges)

        mb = dag.markov_blanket(x)
        nodes = sorted(dag.nodes)
        tag = _var_name("mb", x)

        clauses: List[z3.BoolRef] = []
        mb_vars: Dict[str, z3.BoolRef] = {}

        for n in nodes:
            v = self._bool(f"{tag}_{n}")
            mb_vars[n] = v
            if n in mb:
                clauses.append(v)
            elif n == x:
                clauses.append(z3.Not(v))
            else:
                clauses.append(z3.Not(v))

        # Structural constraints: parents
        for p in dag.parents(x):
            clauses.append(mb_vars[p])

        # Children
        for c in dag.children(x):
            clauses.append(mb_vars[c])
            # Co-parents of children
            for cp in dag.parents(c):
                if cp != x:
                    clauses.append(mb_vars[cp])

        return z3.And(*clauses)

    def encode_markov_blanket_completeness(
        self,
        x: str,
        claimed_mb: Set[str],
        dag: Optional[DAG] = None,
        edges: Optional[Iterable[Tuple[str, str]]] = None,
    ) -> z3.BoolRef:
        """
        Assert that *claimed_mb* is exactly MB(X) – no missing and no
        extra members.
        """
        if dag is None:
            if edges is None:
                raise ValueError("Provide either dag or edges")
            dag = DAG(edges)

        true_mb = dag.markov_blanket(x)
        tag = _var_name("mbc", x)
        clauses: List[z3.BoolRef] = []

        for n in sorted(dag.nodes):
            v = self._bool(f"{tag}_{n}")
            in_claimed = n in claimed_mb
            in_true = n in true_mb
            if in_claimed and in_true:
                clauses.append(v)
            elif not in_claimed and not in_true:
                clauses.append(z3.Not(v))
            elif in_claimed and not in_true:
                # Claimed but wrong → unsatisfiable
                clauses.append(v)
                clauses.append(z3.Not(v))
            else:
                # Missing from claim → unsatisfiable
                clauses.append(v)
                clauses.append(z3.Not(v))

        return z3.And(*clauses)

    # ------------------------------------------------------------------
    # 5. Intervention target encoding
    # ------------------------------------------------------------------

    def encode_intervention(
        self,
        targets: Iterable[str],
        dag: Optional[DAG] = None,
        edges: Optional[Iterable[Tuple[str, str]]] = None,
    ) -> z3.BoolRef:
        """
        Encode the graph-mutilation for do(targets):
        all incoming edges to intervention targets are removed.

        Returns constraints asserting:
        - ``edge(u, t) = False`` for all t ∈ targets, u ∈ parents(t)
        - All other edges remain unchanged.
        """
        if dag is None:
            if edges is None:
                raise ValueError("Provide either dag or edges")
            dag = DAG(edges)

        target_set = set(targets)
        tag = "intv"
        clauses: List[z3.BoolRef] = []

        for u, v in sorted(dag.edge_set):
            edge_var = self._bool(f"{tag}_e_{u}_{v}")
            mutated_var = self._bool(f"{tag}_me_{u}_{v}")
            # Original edge present
            clauses.append(edge_var)
            if v in target_set:
                # Edge removed by intervention
                clauses.append(z3.Not(mutated_var))
            else:
                # Edge preserved
                clauses.append(mutated_var == edge_var)

        # For non-existent edges, mutated graph also has no edge
        all_pairs = {(u, v) for u in dag.nodes for v in dag.nodes if u != v}
        non_edges = all_pairs - dag.edge_set
        for u, v in sorted(non_edges):
            mutated_var = self._bool(f"{tag}_me_{u}_{v}")
            clauses.append(z3.Not(mutated_var))

        return z3.And(*clauses) if clauses else z3.BoolVal(True)

    def encode_intervention_validity(
        self,
        targets: Iterable[str],
        dag: Optional[DAG] = None,
        edges: Optional[Iterable[Tuple[str, str]]] = None,
    ) -> z3.BoolRef:
        """
        Assert that the intervention targets are valid:
        each target must be a node in the DAG.
        """
        if dag is None:
            if edges is None:
                raise ValueError("Provide either dag or edges")
            dag = DAG(edges)

        tag = "intv_valid"
        clauses: List[z3.BoolRef] = []
        for t in targets:
            v = self._bool(f"{tag}_{t}")
            if t in dag.nodes:
                clauses.append(v)
            else:
                clauses.append(z3.Not(v))
                clauses.append(v)  # contradiction → unsat

        return z3.And(*clauses) if clauses else z3.BoolVal(True)

    # ------------------------------------------------------------------
    # 6. Topological ordering
    # ------------------------------------------------------------------

    def encode_topo_order(
        self,
        variables: Iterable[str],
        dag: Optional[DAG] = None,
        edges: Optional[Iterable[Tuple[str, str]]] = None,
    ) -> z3.BoolRef:
        """
        Encode that the list *variables* is a valid topological ordering
        of the DAG.

        Creates integer position variables ``pos_<n>`` and asserts
        ``pos_u < pos_v`` for every edge ``(u, v)``.
        """
        if dag is None:
            if edges is None:
                raise ValueError("Provide either dag or edges")
            dag = DAG(edges)

        var_list = list(variables)
        tag = "topo"
        pos: Dict[str, z3.ArithRef] = {}

        clauses: List[z3.BoolRef] = []

        for idx, n in enumerate(var_list):
            pv = self._int(f"{tag}_pos_{n}")
            pos[n] = pv
            clauses.append(pv == z3.IntVal(idx))

        # Edge constraints: parent before child
        for u, v in dag.edge_set:
            if u in pos and v in pos:
                clauses.append(pos[u] < pos[v])

        # All positions distinct
        all_pos = list(pos.values())
        if len(all_pos) >= 2:
            clauses.append(z3.Distinct(*all_pos))

        return z3.And(*clauses)

    def encode_topo_order_exists(
        self,
        dag: Optional[DAG] = None,
        edges: Optional[Iterable[Tuple[str, str]]] = None,
    ) -> z3.BoolRef:
        """
        Encode that SOME topological ordering exists (i.e. the graph
        is acyclic).

        Creates free integer position variables and asserts the edge
        constraints without fixing values.
        """
        if dag is None:
            if edges is None:
                raise ValueError("Provide either dag or edges")
            dag = DAG(edges)

        tag = "topo_ex"
        nodes = sorted(dag.nodes)
        pos: Dict[str, z3.ArithRef] = {}

        clauses: List[z3.BoolRef] = []
        for n in nodes:
            pv = self._int(f"{tag}_pos_{n}")
            pos[n] = pv
            clauses.append(pv >= z3.IntVal(0))
            clauses.append(pv < z3.IntVal(len(nodes)))

        for u, v in dag.edge_set:
            if u in pos and v in pos:
                clauses.append(pos[u] < pos[v])

        all_pos = list(pos.values())
        if len(all_pos) >= 2:
            clauses.append(z3.Distinct(*all_pos))

        return z3.And(*clauses)

    # ------------------------------------------------------------------
    # 7. Acyclicity encoding
    # ------------------------------------------------------------------

    def encode_acyclicity(
        self,
        dag: Optional[DAG] = None,
        edges: Optional[Iterable[Tuple[str, str]]] = None,
    ) -> z3.BoolRef:
        """
        Assert that the graph is acyclic by requiring existence of a
        topological ordering.
        """
        return self.encode_topo_order_exists(dag=dag, edges=edges)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_variable_count(self) -> int:
        return len(self._var_cache)

    def clear_cache(self) -> None:
        self._var_cache.clear()
