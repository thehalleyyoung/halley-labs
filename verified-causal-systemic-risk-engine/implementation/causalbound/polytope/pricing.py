"""
Pricing Sub-problem
====================

Solves the pricing sub-problem for column generation over a causal polytope.
Exploits the DAG / junction-tree structure via dynamic programming on cliques
to efficiently generate columns with the most negative reduced cost.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Data structures
# ---------------------------------------------------------------------------

class PricingStrategy(Enum):
    EXACT = "exact"
    HEURISTIC = "heuristic"
    RANDOMIZED = "randomized"

    def __init__(self, value: str):
        pass


@dataclass
class PricingResult:
    """Result of a single pricing solve."""
    column: np.ndarray       # full-dimensional point (joint distribution)
    reduced_cost: float
    strategy_used: str
    clique_values: Optional[Dict[FrozenSet[str], np.ndarray]] = None


@dataclass
class JunctionTreeClique:
    """A clique in the junction tree."""
    clique_id: int
    variables: FrozenSet[str]
    separator: FrozenSet[str]  # separator with parent clique
    parent_id: Optional[int] = None
    children_ids: List[int] = field(default_factory=list)
    card: int = 0  # product of variable cardinalities


@dataclass
class JunctionTree:
    """Junction tree built from the moralised DAG."""
    cliques: List[JunctionTreeClique]
    root_id: int = 0

    def leaves(self) -> List[int]:
        return [c.clique_id for c in self.cliques if not c.children_ids]

    def postorder(self) -> List[int]:
        """Return clique ids in postorder (leaves first)."""
        order: List[int] = []
        visited: Set[int] = set()

        def _dfs(cid: int) -> None:
            visited.add(cid)
            for child in self.cliques[cid].children_ids:
                if child not in visited:
                    _dfs(child)
            order.append(cid)

        _dfs(self.root_id)
        return order

    def preorder(self) -> List[int]:
        """Return clique ids in preorder (root first)."""
        order: List[int] = []
        visited: Set[int] = set()

        def _dfs(cid: int) -> None:
            visited.add(cid)
            order.append(cid)
            for child in self.cliques[cid].children_ids:
                if child not in visited:
                    _dfs(child)

        _dfs(self.root_id)
        return order


# ---------------------------------------------------------------------------
#  Pricing sub-problem
# ---------------------------------------------------------------------------

class PricingSubproblem:
    """
    Pricing sub-problem for column generation.

    Given dual variables pi from the master LP, the pricing problem is:

        min   (c - A^T pi)^T x
        s.t.  x in P (the causal polytope)

    where P is the set of valid joint distributions Markov-compatible
    with the DAG.

    The DAG structure is exploited by building a junction tree of the
    moral graph and running DP over its cliques.

    Parameters
    ----------
    dag : DAGSpec
        The (possibly mutilated) DAG.
    c : ndarray
        Full-dimensional objective.
    A_eq : sparse matrix
        Equality constraints.
    b_eq : ndarray
        RHS.
    """

    def __init__(self, dag, c: np.ndarray, A_eq: sparse.spmatrix, b_eq: np.ndarray):
        self.dag = dag
        self.c = c
        self.A_eq = A_eq
        self.b_eq = b_eq
        self._jtree: Optional[JunctionTree] = None
        self._topo = dag.topological_order()
        self._strides = self._compute_strides()
        self._total_vars = self._compute_total_vars()
        self._pricing_cache: Dict[int, PricingResult] = {}
        self._cache_hits: int = 0

        # Build junction tree once
        self._jtree = self._build_junction_tree()

    def price(
        self,
        duals: np.ndarray,
        strategy: PricingStrategy = PricingStrategy.EXACT,
        max_columns: int = 5,
    ) -> List[PricingResult]:
        """
        Run the pricing sub-problem.

        Parameters
        ----------
        duals : ndarray
            Dual variables from the master LP.
        strategy : PricingStrategy
            Which pricing strategy to use.
        max_columns : int
            Maximum number of columns to return.

        Returns
        -------
        List of PricingResult with negative reduced cost.
        """
        # Compute reduced costs:  rc_j = c_j - a_j^T pi
        if sparse.issparse(self.A_eq):
            rc = self.c - self.A_eq.T.dot(duals[:self.A_eq.shape[0]])
        else:
            n_rows = min(len(duals), self.A_eq.shape[0])
            rc = self.c - self.A_eq[:n_rows].T @ duals[:n_rows]

        # Check cache
        cache_key = _hash_duals(duals)
        if cache_key in self._pricing_cache:
            self._cache_hits += 1
            cached = self._pricing_cache[cache_key]
            if cached.reduced_cost < -1e-12:
                return [cached]

        results: List[PricingResult] = []

        if strategy == PricingStrategy.EXACT:
            results = self._exact_pricing(rc, max_columns)
        elif strategy == PricingStrategy.HEURISTIC:
            results = self._heuristic_pricing(rc, max_columns)
        elif strategy == PricingStrategy.RANDOMIZED:
            results = self._randomized_pricing(rc, max_columns)

        # Cache best result
        if results:
            best = min(results, key=lambda r: r.reduced_cost)
            self._pricing_cache[cache_key] = best
            # Limit cache size
            if len(self._pricing_cache) > 1000:
                oldest = next(iter(self._pricing_cache))
                del self._pricing_cache[oldest]

        return results

    # ------------------------------------------------------------------
    #  Exact pricing via junction-tree DP
    # ------------------------------------------------------------------

    def _exact_pricing(self, rc: np.ndarray, max_columns: int) -> List[PricingResult]:
        """
        Exact pricing by DP on the junction tree.

        For each clique, we compute the minimum reduced-cost assignment
        consistent with the separator assignments from the parent clique.
        """
        if self._jtree is None or not self._jtree.cliques:
            return self._fallback_pricing(rc, max_columns)

        results: List[PricingResult] = []

        # Phase 1: upward pass (collect) - compute min cost per separator config
        clique_tables: Dict[int, np.ndarray] = {}
        clique_argmins: Dict[int, np.ndarray] = {}

        for cid in self._jtree.postorder():
            clique = self._jtree.cliques[cid]
            clique_vars = sorted(clique.variables)

            # Compute local cost table: for each assignment of clique vars,
            # the reduced cost of that clique's contribution
            table_size = 1
            for v in clique_vars:
                table_size *= self.dag.card[v]

            cost_table = np.zeros(table_size, dtype=np.float64)
            self._fill_clique_cost_table(cost_table, clique_vars, rc)

            # Add children's messages (min over child-only variables)
            for child_id in clique.children_ids:
                child_clique = self._jtree.cliques[child_id]
                child_msg = clique_tables.get(child_id)
                if child_msg is not None:
                    cost_table = self._absorb_child_message(
                        cost_table, clique_vars,
                        child_msg, sorted(child_clique.separator),
                    )

            clique_tables[cid] = cost_table

            # Compute message to parent: minimise over non-separator variables
            if clique.parent_id is not None:
                sep_vars = sorted(clique.separator)
                message = self._marginalise_min(cost_table, clique_vars, sep_vars)
                clique_tables[cid] = cost_table  # keep full table
                # Store message under a special key
                clique_tables[(-1, cid)] = message

        # Phase 2: downward pass (distribute) - extract optimal assignment
        assignment = self._extract_optimal_assignment(clique_tables)

        # Build the column (joint distribution point-mass at assignment)
        column = np.zeros(self._total_vars, dtype=np.float64)
        flat_idx = self._assignment_to_flat(assignment)
        if 0 <= flat_idx < self._total_vars:
            column[flat_idx] = 1.0
        reduced_cost = float(rc @ column)

        results.append(PricingResult(
            column=column,
            reduced_cost=reduced_cost,
            strategy_used="exact_jt",
        ))

        # Generate additional columns by perturbing
        if max_columns > 1:
            extra = self._generate_neighbor_columns(assignment, rc, max_columns - 1)
            results.extend(extra)

        return [r for r in results if r.reduced_cost < -1e-12]

    def _fill_clique_cost_table(
        self,
        table: np.ndarray,
        clique_vars: List[str],
        rc: np.ndarray,
    ) -> None:
        """
        Fill the cost table for a clique by summing reduced costs of all
        joint assignments that match each clique configuration.
        """
        clique_card = [self.dag.card[v] for v in clique_vars]
        clique_size = len(table)

        # For each clique assignment, sum rc over all compatible full assignments
        clique_strides = {}
        s = 1
        for v in reversed(clique_vars):
            clique_strides[v] = s
            s *= self.dag.card[v]

        for ci in range(clique_size):
            # Decode clique assignment
            clique_assign: Dict[str, int] = {}
            rem = ci
            for v in clique_vars:
                card = self.dag.card[v]
                stride = clique_strides[v]
                clique_assign[v] = (rem // stride) % card

            # Sum reduced costs over all full assignments consistent with clique_assign
            other_vars = [v for v in self._topo if v not in clique_vars]
            if not other_vars:
                flat = self._assignment_to_flat(clique_assign)
                if 0 <= flat < len(rc):
                    table[ci] = rc[flat]
                continue

            # Enumerate over other variables
            other_card = [self.dag.card[v] for v in other_vars]
            n_other = 1
            for c in other_card:
                n_other *= c

            total_rc = 0.0
            for oi in range(n_other):
                full_assign = dict(clique_assign)
                rem_o = oi
                for j, v in enumerate(other_vars):
                    c = other_card[j]
                    full_assign[v] = rem_o % c
                    rem_o //= c
                flat = self._assignment_to_flat(full_assign)
                if 0 <= flat < len(rc):
                    total_rc += rc[flat]

            table[ci] = total_rc

    def _absorb_child_message(
        self,
        parent_table: np.ndarray,
        parent_vars: List[str],
        child_message: np.ndarray,
        separator_vars: List[str],
    ) -> np.ndarray:
        """Add child's message (indexed by separator) to parent table."""
        if len(child_message) == 0:
            return parent_table

        result = parent_table.copy()
        parent_card = [self.dag.card[v] for v in parent_vars]
        sep_card = [self.dag.card[v] for v in separator_vars]

        parent_strides = {}
        s = 1
        for v in reversed(parent_vars):
            parent_strides[v] = s
            s *= self.dag.card[v]

        sep_strides = {}
        s = 1
        for v in reversed(separator_vars):
            sep_strides[v] = s
            s *= self.dag.card[v]

        for pi in range(len(parent_table)):
            # Decode parent assignment for separator vars
            rem = pi
            sep_idx = 0
            for v in parent_vars:
                card = self.dag.card[v]
                stride = parent_strides[v]
                val = (rem // stride) % card
                if v in sep_strides:
                    sep_idx += val * sep_strides[v]

            if sep_idx < len(child_message):
                result[pi] += child_message[sep_idx]

        return result

    def _marginalise_min(
        self,
        table: np.ndarray,
        all_vars: List[str],
        keep_vars: List[str],
    ) -> np.ndarray:
        """Minimise table over variables not in keep_vars."""
        if not keep_vars:
            return np.array([np.min(table)], dtype=np.float64)

        keep_card = [self.dag.card[v] for v in keep_vars]
        keep_size = 1
        for c in keep_card:
            keep_size *= c

        keep_strides = {}
        s = 1
        for v in reversed(keep_vars):
            keep_strides[v] = s
            s *= self.dag.card[v]

        all_strides = {}
        s = 1
        for v in reversed(all_vars):
            all_strides[v] = s
            s *= self.dag.card[v]

        message = np.full(keep_size, float("inf"), dtype=np.float64)

        for ai in range(len(table)):
            # Map full index to keep index
            ki = 0
            rem = ai
            for v in all_vars:
                card = self.dag.card[v]
                stride = all_strides[v]
                val = (rem // stride) % card
                if v in keep_strides:
                    ki += val * keep_strides[v]

            message[ki] = min(message[ki], table[ai])

        return message

    def _extract_optimal_assignment(
        self,
        clique_tables: Dict,
    ) -> Dict[str, int]:
        """Extract the optimal variable assignment from DP tables."""
        assignment: Dict[str, int] = {}

        if self._jtree is None:
            return assignment

        for cid in self._jtree.preorder():
            clique = self._jtree.cliques[cid]
            clique_vars = sorted(clique.variables)

            table = clique_tables.get(cid)
            if table is None:
                continue

            # Find assignment of unset variables that minimises cost
            # consistent with already-fixed variables
            clique_strides = {}
            s = 1
            for v in reversed(clique_vars):
                clique_strides[v] = s
                s *= self.dag.card[v]

            best_cost = float("inf")
            best_idx = 0

            for ci in range(len(table)):
                # Check consistency with fixed vars
                consistent = True
                rem = ci
                for v in clique_vars:
                    card = self.dag.card[v]
                    stride = clique_strides[v]
                    val = (rem // stride) % card
                    if v in assignment and assignment[v] != val:
                        consistent = False
                        break

                if consistent and table[ci] < best_cost:
                    best_cost = table[ci]
                    best_idx = ci

            # Decode and fix
            rem = best_idx
            for v in clique_vars:
                if v not in assignment:
                    card = self.dag.card[v]
                    stride = clique_strides[v]
                    assignment[v] = (rem // stride) % card

        # Fill any remaining variables greedily
        for v in self._topo:
            if v not in assignment:
                assignment[v] = 0

        return assignment

    # ------------------------------------------------------------------
    #  Heuristic pricing
    # ------------------------------------------------------------------

    def _heuristic_pricing(self, rc: np.ndarray, max_columns: int) -> List[PricingResult]:
        """
        Heuristic pricing: greedily build a column by fixing variables
        in topological order, choosing the value that locally minimises
        reduced cost.
        """
        results: List[PricingResult] = []

        for trial in range(max_columns):
            assignment: Dict[str, int] = {}

            for node in self._topo:
                best_val = 0
                best_cost = float("inf")
                card = self.dag.card[node]

                for val in range(card):
                    assignment[node] = val
                    # Compute partial reduced cost
                    partial_rc = self._partial_reduced_cost(assignment, rc)
                    if partial_rc < best_cost:
                        best_cost = partial_rc
                        best_val = val

                # Add some randomness for diversity on later trials
                if trial > 0:
                    rng = np.random.default_rng(42 + trial)
                    if rng.random() < 0.2:
                        best_val = rng.integers(0, card)

                assignment[node] = best_val

            column = np.zeros(self._total_vars, dtype=np.float64)
            flat_idx = self._assignment_to_flat(assignment)
            if 0 <= flat_idx < self._total_vars:
                column[flat_idx] = 1.0
            reduced_cost = float(rc @ column)

            results.append(PricingResult(
                column=column,
                reduced_cost=reduced_cost,
                strategy_used="heuristic",
            ))

        return [r for r in results if r.reduced_cost < -1e-12]

    def _partial_reduced_cost(self, assignment: Dict[str, int], rc: np.ndarray) -> float:
        """
        Compute the reduced cost contribution from the fixed variables
        in a partial assignment.
        """
        fixed_vars = list(assignment.keys())
        unfixed = [v for v in self._topo if v not in assignment]

        if not unfixed:
            flat = self._assignment_to_flat(assignment)
            if 0 <= flat < len(rc):
                return rc[flat]
            return float("inf")

        # Sum minimum rc over all completions
        unfixed_cards = [self.dag.card[v] for v in unfixed]
        n_completions = 1
        for c in unfixed_cards:
            n_completions *= c

        # For efficiency, cap enumeration
        if n_completions > 10000:
            # Just evaluate a sample
            rng = np.random.default_rng(hash(tuple(sorted(assignment.items()))) % (2**31))
            best = float("inf")
            for _ in range(min(100, n_completions)):
                full = dict(assignment)
                for v, c in zip(unfixed, unfixed_cards):
                    full[v] = rng.integers(0, c)
                flat = self._assignment_to_flat(full)
                if 0 <= flat < len(rc):
                    best = min(best, rc[flat])
            return best

        best = float("inf")
        for ci in range(n_completions):
            full = dict(assignment)
            rem = ci
            for v, c in zip(unfixed, unfixed_cards):
                full[v] = rem % c
                rem //= c
            flat = self._assignment_to_flat(full)
            if 0 <= flat < len(rc):
                best = min(best, rc[flat])

        return best

    # ------------------------------------------------------------------
    #  Randomised pricing
    # ------------------------------------------------------------------

    def _randomized_pricing(self, rc: np.ndarray, max_columns: int) -> List[PricingResult]:
        """
        Randomised pricing: sample columns with probability proportional
        to exp(-rc / temperature).
        """
        results: List[PricingResult] = []
        temperature = max(0.01, float(np.std(rc)))

        rng = np.random.default_rng()

        for _ in range(max_columns * 3):
            # Sample assignment
            assignment: Dict[str, int] = {}
            for node in self._topo:
                card = self.dag.card[node]
                # Compute conditional costs for each value
                costs = np.zeros(card, dtype=np.float64)
                for val in range(card):
                    assignment[node] = val
                    flat = self._assignment_to_flat(assignment)
                    if 0 <= flat < len(rc):
                        costs[val] = rc[flat]
                    else:
                        costs[val] = 0.0

                # Softmin sampling
                log_probs = -costs / temperature
                log_probs -= np.max(log_probs)
                probs = np.exp(log_probs)
                probs /= probs.sum() + 1e-300
                assignment[node] = int(rng.choice(card, p=probs))

            column = np.zeros(self._total_vars, dtype=np.float64)
            flat_idx = self._assignment_to_flat(assignment)
            if 0 <= flat_idx < self._total_vars:
                column[flat_idx] = 1.0
            reduced_cost = float(rc @ column)

            results.append(PricingResult(
                column=column,
                reduced_cost=reduced_cost,
                strategy_used="randomized",
            ))

        # Keep best
        results.sort(key=lambda r: r.reduced_cost)
        results = results[:max_columns]
        return [r for r in results if r.reduced_cost < -1e-12]

    # ------------------------------------------------------------------
    #  Fallback pricing (LP-based)
    # ------------------------------------------------------------------

    def _fallback_pricing(self, rc: np.ndarray, max_columns: int) -> List[PricingResult]:
        """
        Fallback pricing when junction tree is unavailable:
        solve a small LP or enumerate vertices.
        """
        results: List[PricingResult] = []
        n = self._total_vars

        if n <= 1000:
            # Small enough to enumerate all vertices
            best_idx = np.argmin(rc)
            column = np.zeros(n, dtype=np.float64)
            column[best_idx] = 1.0
            results.append(PricingResult(
                column=column,
                reduced_cost=float(rc[best_idx]),
                strategy_used="enumerate",
            ))

            # Also add k-best
            sorted_indices = np.argsort(rc)
            for i in range(1, min(max_columns, len(sorted_indices))):
                idx = sorted_indices[i]
                if rc[idx] < -1e-12:
                    col = np.zeros(n, dtype=np.float64)
                    col[idx] = 1.0
                    results.append(PricingResult(
                        column=col,
                        reduced_cost=float(rc[idx]),
                        strategy_used="enumerate",
                    ))
        else:
            # Solve pricing as LP over simplex
            bounds = [(0.0, 1.0)] * n
            A_eq_norm = np.ones((1, n), dtype=np.float64)
            b_eq_norm = np.array([1.0], dtype=np.float64)

            res = linprog(
                c=rc,
                A_eq=A_eq_norm,
                b_eq=b_eq_norm,
                bounds=bounds,
                method="highs",
            )
            if res.success:
                results.append(PricingResult(
                    column=res.x,
                    reduced_cost=float(res.fun),
                    strategy_used="lp_simplex",
                ))

        return [r for r in results if r.reduced_cost < -1e-12]

    # ------------------------------------------------------------------
    #  Generate neighbouring columns
    # ------------------------------------------------------------------

    def _generate_neighbor_columns(
        self,
        assignment: Dict[str, int],
        rc: np.ndarray,
        count: int,
    ) -> List[PricingResult]:
        """Generate columns by flipping single variables in the assignment."""
        results: List[PricingResult] = []

        for node in self._topo:
            if len(results) >= count:
                break
            orig_val = assignment.get(node, 0)
            card = self.dag.card[node]

            for v in range(card):
                if v == orig_val:
                    continue
                new_assign = dict(assignment)
                new_assign[node] = v

                column = np.zeros(self._total_vars, dtype=np.float64)
                flat_idx = self._assignment_to_flat(new_assign)
                if 0 <= flat_idx < self._total_vars:
                    column[flat_idx] = 1.0
                reduced_cost = float(rc @ column)

                if reduced_cost < -1e-12:
                    results.append(PricingResult(
                        column=column,
                        reduced_cost=reduced_cost,
                        strategy_used="neighbor",
                    ))

                if len(results) >= count:
                    break

        results.sort(key=lambda r: r.reduced_cost)
        return results[:count]

    # ------------------------------------------------------------------
    #  Junction tree construction
    # ------------------------------------------------------------------

    def _build_junction_tree(self) -> JunctionTree:
        """
        Build a junction tree from the moral graph of the DAG.

        Steps:
        1. Moralise the DAG.
        2. Triangulate by maximum-cardinality search.
        3. Extract maximal cliques.
        4. Build junction tree by maximum-weight spanning tree on clique graph.
        """
        moral_adj = self.dag.moralize()
        triangulated_adj = self._triangulate(moral_adj)
        cliques = self._find_maximal_cliques(triangulated_adj)

        if not cliques:
            return JunctionTree(cliques=[], root_id=0)

        jt_cliques, root_id = self._build_clique_tree(cliques)
        return JunctionTree(cliques=jt_cliques, root_id=root_id)

    def _triangulate(self, adj: Dict[str, set]) -> Dict[str, set]:
        """
        Triangulate the moral graph using the maximum-cardinality-search
        elimination ordering, then adding fill edges.
        """
        nodes = list(adj.keys())
        if not nodes:
            return adj

        result_adj: Dict[str, set] = {n: set(adj[n]) for n in nodes}
        eliminated: Set[str] = set()

        # Maximum cardinality search for elimination order
        order: List[str] = []
        weight: Dict[str, int] = {n: 0 for n in nodes}

        for _ in range(len(nodes)):
            # Pick node with maximum weight among un-eliminated
            best_node = None
            best_w = -1
            for n in nodes:
                if n not in eliminated and weight[n] > best_w:
                    best_w = weight[n]
                    best_node = n
            if best_node is None:
                break

            order.append(best_node)
            eliminated.add(best_node)

            # Update weights
            for nbr in result_adj[best_node]:
                if nbr not in eliminated:
                    weight[nbr] += 1

        # Eliminate in reverse order, adding fill edges
        eliminated.clear()
        for node in reversed(order):
            active_nbrs = [n for n in result_adj[node] if n not in eliminated]
            # Make active neighbours a clique (fill edges)
            for i in range(len(active_nbrs)):
                for j in range(i + 1, len(active_nbrs)):
                    u, v = active_nbrs[i], active_nbrs[j]
                    result_adj[u].add(v)
                    result_adj[v].add(u)
            eliminated.add(node)

        return result_adj

    def _find_maximal_cliques(self, adj: Dict[str, set]) -> List[FrozenSet[str]]:
        """Find all maximal cliques in the triangulated graph (Bron-Kerbosch)."""
        nodes = sorted(adj.keys())
        cliques: List[FrozenSet[str]] = []

        def _bron_kerbosch(R: Set[str], P: Set[str], X: Set[str]) -> None:
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
            # Choose pivot
            pivot = max(P | X, key=lambda n: len(adj[n] & P), default=None)
            if pivot is None:
                return
            candidates = P - adj[pivot]
            for v in list(candidates):
                _bron_kerbosch(
                    R | {v},
                    P & adj[v],
                    X & adj[v],
                )
                P.remove(v)
                X.add(v)

        _bron_kerbosch(set(), set(nodes), set())

        # If no cliques found, treat each edge as a clique
        if not cliques:
            for n in nodes:
                cliques.append(frozenset({n}))

        return cliques

    def _build_clique_tree(
        self,
        cliques: List[FrozenSet[str]],
    ) -> Tuple[List[JunctionTreeClique], int]:
        """
        Build a junction tree from maximal cliques by finding the maximum
        weight spanning tree of the clique intersection graph, where
        edge weight = |separator|.
        """
        n = len(cliques)
        if n == 0:
            return [], 0

        if n == 1:
            jt_c = JunctionTreeClique(
                clique_id=0,
                variables=cliques[0],
                separator=frozenset(),
                card=self._clique_card(cliques[0]),
            )
            return [jt_c], 0

        # Build edge weights
        edges: List[Tuple[int, int, int]] = []
        for i in range(n):
            for j in range(i + 1, n):
                sep = cliques[i] & cliques[j]
                if sep:
                    edges.append((i, j, len(sep)))

        # Kruskal's for maximum spanning tree
        edges.sort(key=lambda e: -e[2])
        parent_uf = list(range(n))

        def find(x: int) -> int:
            while parent_uf[x] != x:
                parent_uf[x] = parent_uf[parent_uf[x]]
                x = parent_uf[x]
            return x

        tree_edges: List[Tuple[int, int, FrozenSet[str]]] = []
        for i, j, w in edges:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent_uf[ri] = rj
                sep = cliques[i] & cliques[j]
                tree_edges.append((i, j, sep))
                if len(tree_edges) == n - 1:
                    break

        # Handle disconnected components
        for i in range(1, n):
            ri = find(i)
            r0 = find(0)
            if ri != r0:
                parent_uf[ri] = r0
                tree_edges.append((0, i, frozenset()))

        # Build adjacency for the tree
        tree_adj: Dict[int, List[Tuple[int, FrozenSet[str]]]] = {i: [] for i in range(n)}
        for i, j, sep in tree_edges:
            tree_adj[i].append((j, sep))
            tree_adj[j].append((i, sep))

        # Root at 0 and build parent-child structure
        jt_cliques: List[JunctionTreeClique] = []
        for i in range(n):
            jt_cliques.append(JunctionTreeClique(
                clique_id=i,
                variables=cliques[i],
                separator=frozenset(),
                card=self._clique_card(cliques[i]),
            ))

        # BFS to set parent/children
        visited = {0}
        queue = [0]
        while queue:
            current = queue.pop(0)
            for nbr, sep in tree_adj[current]:
                if nbr not in visited:
                    visited.add(nbr)
                    jt_cliques[nbr].parent_id = current
                    jt_cliques[nbr].separator = sep
                    jt_cliques[current].children_ids.append(nbr)
                    queue.append(nbr)

        return jt_cliques, 0

    def _clique_card(self, variables: FrozenSet[str]) -> int:
        card = 1
        for v in variables:
            card *= self.dag.card[v]
        return card

    # ------------------------------------------------------------------
    #  Utility
    # ------------------------------------------------------------------

    def _compute_strides(self) -> Dict[str, int]:
        strides: Dict[str, int] = {}
        s = 1
        for node in reversed(self._topo):
            strides[node] = s
            s *= self.dag.card[node]
        return strides

    def _compute_total_vars(self) -> int:
        total = 1
        for node in self.dag.nodes:
            total *= self.dag.card[node]
        return total

    def _assignment_to_flat(self, assignment: Dict[str, int]) -> int:
        idx = 0
        for node in self._topo:
            val = assignment.get(node, 0)
            idx += val * self._strides[node]
        return idx


def _hash_duals(duals: np.ndarray, precision: int = 6) -> int:
    """Hash dual variables for caching (round to avoid float issues)."""
    rounded = np.round(duals, precision)
    return hash(rounded.tobytes())
