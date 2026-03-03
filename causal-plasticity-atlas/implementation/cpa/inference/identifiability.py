"""Causal effect identifiability analysis.

Implements the ID algorithm (Tian & Pearl, 2002), hedge criterion,
c-component factorization, generalized adjustment criterion, and
adjustment-set search for determining whether a causal effect can be
uniquely computed from observational data in semi-Markovian models.

Semi-Markovian models are represented by a directed adjacency matrix
(for directed edges) plus a symmetric bidirected-edge matrix (for
latent common causes).
"""

from __future__ import annotations

import itertools
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------
# Data class
# ---------------------------------------------------------------

@dataclass
class IdentifiabilityResult:
    """Result of an identifiability check."""

    identifiable: bool
    estimand: Optional[str] = None
    hedge: Optional[tuple[set[int], set[int]]] = None
    c_components: list[set[int]] = field(default_factory=list)
    derivation_steps: list[str] = field(default_factory=list)


# ---------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------

def _parents_of(adj: NDArray, j: int) -> List[int]:
    return list(np.nonzero(adj[:, j])[0])


def _children_of(adj: NDArray, i: int) -> List[int]:
    return list(np.nonzero(adj[i, :])[0])


def _ancestors_of(adj: NDArray, nodes: Set[int]) -> Set[int]:
    p = adj.shape[0]
    result: set[int] = set()
    stack = list(nodes)
    while stack:
        n = stack.pop()
        for par in range(p):
            if adj[par, n] != 0 and par not in result:
                result.add(par)
                stack.append(par)
    return result


def _descendants_of(adj: NDArray, nodes: Set[int]) -> Set[int]:
    p = adj.shape[0]
    result: set[int] = set()
    stack = list(nodes)
    while stack:
        n = stack.pop()
        for ch in range(p):
            if adj[n, ch] != 0 and ch not in result:
                result.add(ch)
                stack.append(ch)
    return result


def _topological_sort(adj: NDArray) -> List[int]:
    p = adj.shape[0]
    binary = (adj != 0).astype(int)
    in_deg = binary.sum(axis=0).tolist()
    queue: deque[int] = deque(i for i in range(p) if in_deg[i] == 0)
    order: list[int] = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for ch in range(p):
            if binary[node, ch]:
                in_deg[ch] -= 1
                if in_deg[ch] == 0:
                    queue.append(ch)
    if len(order) != p:
        raise ValueError("Graph contains a cycle")
    return order


def _topological_sort_subset(adj: NDArray, nodes: Set[int]) -> List[int]:
    """Topological sort restricted to a subset of nodes."""
    if not nodes:
        return []
    idx_list = sorted(nodes)
    sub = adj[np.ix_(idx_list, idx_list)]
    sub_order = _topological_sort(sub)
    return [idx_list[i] for i in sub_order]


def _d_separated(
    adj: NDArray, x: Set[int], y: Set[int], z: Set[int]
) -> bool:
    """Bayes-Ball d-separation test."""
    if x & y:
        return False
    p = adj.shape[0]
    visited: set[tuple[int, str]] = set()
    queue: deque[tuple[int, str]] = deque()
    reachable: set[int] = set()
    for s in x:
        queue.append((s, "up"))
    while queue:
        node, direction = queue.popleft()
        if (node, direction) in visited:
            continue
        visited.add((node, direction))
        if node not in x:
            reachable.add(node)
        if direction == "up" and node not in z:
            for par in _parents_of(adj, node):
                if (par, "up") not in visited:
                    queue.append((par, "up"))
            for ch in _children_of(adj, node):
                if (ch, "down") not in visited:
                    queue.append((ch, "down"))
        elif direction == "down":
            if node not in z:
                for ch in _children_of(adj, node):
                    if (ch, "down") not in visited:
                        queue.append((ch, "down"))
            if node in z:
                for par in _parents_of(adj, node):
                    if (par, "up") not in visited:
                        queue.append((par, "up"))
    return len(reachable & y) == 0


def _subgraph(adj: NDArray, nodes: Set[int]) -> Tuple[NDArray, List[int]]:
    """Extract induced subgraph on *nodes*. Returns (sub_adj, sorted_nodes)."""
    idx = sorted(nodes)
    return adj[np.ix_(idx, idx)].copy(), idx


def _fmt(s: Set[int]) -> str:
    return "{" + ", ".join(str(x) for x in sorted(s)) + "}"


# ---------------------------------------------------------------
# IdentifiabilityChecker
# ---------------------------------------------------------------

class IdentifiabilityChecker:
    """Check identifiability of causal effects in semi-Markovian models.

    A semi-Markovian model is specified by:
    - ``graph``: directed adjacency matrix (``adj[i,j] != 0`` ⇒ i → j).
    - Optionally a symmetric bidirected-edge matrix encoding latent
      common causes (``bi[i,j] = bi[j,i] = 1`` ⇒ i ↔ j).

    If no bidirected edges are given, the model is Markovian and all
    effects are identifiable via the truncated factorization formula.

    Parameters
    ----------
    verbose : bool
        Whether to log intermediate steps.
    """

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose
        self._steps: list[str] = []

    # =================================================================
    # Public API
    # =================================================================

    def is_identifiable(
        self,
        graph: NDArray[np.int_],
        treatment: set[int],
        outcome: set[int],
        *,
        bidirected: Optional[NDArray[np.int_]] = None,
    ) -> IdentifiabilityResult:
        """Determine whether P(outcome | do(treatment)) is identifiable.

        Runs the full ID algorithm.  If identifiable, returns the
        estimand expression; otherwise returns a hedge witnessing
        non-identifiability.

        Parameters
        ----------
        graph : ndarray
            Directed adjacency matrix.
        treatment : set of int
        outcome : set of int
        bidirected : ndarray, optional
            Symmetric matrix of bidirected (latent) edges.  If None,
            assumes a fully Markovian model.

        Returns
        -------
        IdentifiabilityResult
        """
        graph = np.asarray(graph, dtype=np.float64)
        p = graph.shape[0]

        if bidirected is None:
            bidirected = np.zeros((p, p), dtype=np.float64)
        else:
            bidirected = np.asarray(bidirected, dtype=np.float64)

        self._steps = []

        all_vars = set(range(p))
        c_comps = self._find_c_components(graph, bidirected, all_vars)

        result = self.id_algorithm(
            graph, outcome, treatment, f"P({_fmt(all_vars)})",
            bidirected=bidirected,
            full_set=all_vars,
        )

        result.c_components = c_comps
        return result

    # =================================================================
    # ID Algorithm (Tian & Pearl, 2002)
    # =================================================================

    def id_algorithm(
        self,
        graph: NDArray[np.int_],
        target: set[int],
        intervention: set[int],
        probability: str,
        *,
        bidirected: Optional[NDArray[np.int_]] = None,
        full_set: Optional[set[int]] = None,
    ) -> IdentifiabilityResult:
        """Run the full ID algorithm.

        Recursive algorithm for identifying P(Y | do(X)) from P(V).

        Parameters
        ----------
        graph : ndarray
            Directed adjacency matrix.
        target : set of int
            Y — outcome variables.
        intervention : set of int
            X — intervention variables.
        probability : str
            Current probability expression.
        bidirected : ndarray, optional
            Bidirected edge matrix.
        full_set : set of int, optional
            Full variable set V.

        Returns
        -------
        IdentifiabilityResult
        """
        graph = np.asarray(graph, dtype=np.float64)
        p = graph.shape[0]
        V = full_set if full_set is not None else set(range(p))

        if bidirected is None:
            bidirected = np.zeros((p, p), dtype=np.float64)

        steps: list[str] = []

        # ------ Line 1: if X = ∅ ------
        if not intervention:
            # P(Y | do(∅)) = Σ_{V\Y} P(V)
            summed = V - target
            if summed:
                expr = f"Σ_{{{_fmt(summed)}}} {probability}"
            else:
                expr = probability
            steps.append(f"ID Line 1: X=∅ → {expr}")
            return IdentifiabilityResult(
                identifiable=True,
                estimand=expr,
                derivation_steps=steps,
            )

        # ------ Line 2: An(Y)_G restriction ------
        an_y = _ancestors_of(graph, target) | target
        non_an = V - an_y
        if non_an:
            steps.append(
                f"ID Line 2: restrict to An(Y)={_fmt(an_y)}; "
                f"remove {_fmt(non_an)}"
            )
            sub_idx = sorted(an_y)
            sub_graph = graph[np.ix_(sub_idx, sub_idx)]
            sub_bi = bidirected[np.ix_(sub_idx, sub_idx)]
            idx_map = {old: new for new, old in enumerate(sub_idx)}
            rev_map = {new: old for old, new in idx_map.items()}
            new_target = {idx_map[v] for v in target if v in idx_map}
            new_interv = {idx_map[v] for v in intervention if v in idx_map}
            new_V = set(range(len(sub_idx)))

            sub_result = self.id_algorithm(
                sub_graph, new_target, new_interv,
                probability,
                bidirected=sub_bi,
                full_set=new_V,
            )
            sub_result.derivation_steps = steps + sub_result.derivation_steps
            return sub_result

        # ------ Line 3: (V \ An(Y)_{G_{\bar X}}) check ------
        g_bar_x = graph.copy()
        for node in intervention:
            if node < p:
                g_bar_x[:, node] = 0

        an_y_gbarx = _ancestors_of(g_bar_x, target) | target
        W = (V - intervention) - an_y_gbarx
        if W:
            steps.append(
                f"ID Line 3: W={_fmt(W)} not ancestors of Y in G_bar_X; "
                f"recurse with do(X ∪ W)"
            )
            return self.id_algorithm(
                graph, target, intervention - W, probability,
                bidirected=bidirected, full_set=V,
            )

        # ------ Line 4: c-component decomposition of G[V \ X] ------
        remaining = V - intervention
        c_comps_remaining = self._find_c_components(
            graph, bidirected, remaining
        )

        if len(c_comps_remaining) > 1:
            steps.append(
                f"ID Line 4: {len(c_comps_remaining)} c-components "
                f"in G[V\\X]"
            )
            # P(Y | do(X)) = Σ_{V\(Y∪X)} Π_i P(S_i | do(V\S_i))
            parts: list[str] = []
            for comp in c_comps_remaining:
                comp_result = self._id_single_component(
                    graph, bidirected, comp, V, probability, steps
                )
                if not comp_result.identifiable:
                    return comp_result
                parts.append(comp_result.estimand or "?")

            product = " × ".join(parts)
            summed = V - (target | intervention)
            if summed:
                expr = f"Σ_{{{_fmt(summed)}}} [{product}]"
            else:
                expr = product
            steps.append(f"ID Line 4 result: {expr}")
            return IdentifiabilityResult(
                identifiable=True,
                estimand=expr,
                derivation_steps=steps,
            )

        # ------ Single c-component ------
        # Line 5 and 6
        S = c_comps_remaining[0] if c_comps_remaining else remaining

        # C-components of G (full graph)
        c_comps_full = self._find_c_components(graph, bidirected, V)

        # Line 5: if C(G) = {V}, fail (hedge found)
        if len(c_comps_full) == 1 and c_comps_full[0] == V:
            steps.append(
                f"ID Line 5: single c-component = V; "
                f"hedge found → NOT identifiable"
            )
            hedge = self._construct_hedge(graph, bidirected, intervention, target, V)
            return IdentifiabilityResult(
                identifiable=False,
                hedge=hedge,
                derivation_steps=steps,
            )

        # Line 6: S ∈ S_i for some S_i ∈ C(G)
        containing_comp = None
        for comp in c_comps_full:
            if S <= comp:
                containing_comp = comp
                break

        if containing_comp is not None and containing_comp != V:
            steps.append(
                f"ID Line 6: S={_fmt(S)} ⊆ S_i={_fmt(containing_comp)}"
            )
            return self._id_line6(
                graph, bidirected, target, S, containing_comp, V,
                probability, steps
            )

        # Line 7: S not contained in any single c-component → hedge
        steps.append(
            f"ID Line 7: S={_fmt(S)} spans multiple c-components → "
            f"hedge → NOT identifiable"
        )
        hedge = self._construct_hedge(graph, bidirected, intervention, target, V)
        return IdentifiabilityResult(
            identifiable=False,
            hedge=hedge,
            derivation_steps=steps,
        )

    # =================================================================
    # ID subroutines
    # =================================================================

    def _id_single_component(
        self,
        graph: NDArray,
        bidirected: NDArray,
        comp: Set[int],
        V: Set[int],
        probability: str,
        parent_steps: list[str],
    ) -> IdentifiabilityResult:
        """Identify P(S_i | do(V \\ S_i)) for a single c-component."""
        intervention = V - comp

        c_comps_full = self._find_c_components(graph, bidirected, V)

        # Check if comp is a subset of some c-component of G
        containing = None
        for cc in c_comps_full:
            if comp <= cc:
                containing = cc
                break

        if containing is None:
            # comp equals a c-component of G[V\X], identify directly
            return self._identify_component_directly(
                graph, comp, V, probability
            )

        if containing == V and len(c_comps_full) == 1:
            # Hedge
            return IdentifiabilityResult(
                identifiable=False,
                hedge=(V, comp),
            )

        if containing != V:
            return self._id_line6(
                graph, bidirected, comp, comp, containing, V,
                probability, parent_steps
            )

        return self._identify_component_directly(graph, comp, V, probability)

    def _identify_component_directly(
        self,
        graph: NDArray,
        comp: Set[int],
        V: Set[int],
        probability: str,
    ) -> IdentifiabilityResult:
        """Direct identification of a c-component factor."""
        p = graph.shape[0]
        try:
            topo = _topological_sort(graph)
        except ValueError:
            topo = sorted(V)

        # P(S_i) = Π_{v ∈ S_i} P(v | v^{(π−1)})
        factors: list[str] = []
        for v in sorted(comp):
            # Predecessors in topological order
            v_idx = topo.index(v) if v in topo else 0
            predecessors = set(topo[:v_idx])
            if predecessors:
                factors.append(f"P(x{v} | {_fmt(predecessors)})")
            else:
                factors.append(f"P(x{v})")

        expr = " × ".join(factors) if factors else probability
        return IdentifiabilityResult(
            identifiable=True,
            estimand=expr,
        )

    def _id_line6(
        self,
        graph: NDArray,
        bidirected: NDArray,
        target: Set[int],
        S: Set[int],
        S_prime: Set[int],
        V: Set[int],
        probability: str,
        parent_steps: list[str],
    ) -> IdentifiabilityResult:
        """ID algorithm Line 6 handling."""
        p = graph.shape[0]
        try:
            topo = _topological_sort(graph)
        except ValueError:
            topo = sorted(V)

        # Construct P(S') = Π_{v ∈ S'} P(v | v^{(π-1)} ∩ S')
        factors: list[str] = []
        s_prime_topo = [v for v in topo if v in S_prime]
        for idx, v in enumerate(s_prime_topo):
            preds = set(s_prime_topo[:idx])
            if preds:
                factors.append(f"P(x{v} | {_fmt(preds)})")
            else:
                factors.append(f"P(x{v})")

        p_s_prime = " × ".join(factors) if factors else "P(?)"

        # Recurse on subproblem within S'
        sub_idx = sorted(S_prime)
        sub_graph = graph[np.ix_(sub_idx, sub_idx)]
        sub_bi = bidirected[np.ix_(sub_idx, sub_idx)]
        idx_map = {old: new for new, old in enumerate(sub_idx)}

        new_target = {idx_map[v] for v in target if v in idx_map}
        new_interv = {idx_map[v] for v in (S_prime - S) if v in idx_map}
        new_V = set(range(len(sub_idx)))

        sub_result = self.id_algorithm(
            sub_graph, new_target, new_interv,
            p_s_prime,
            bidirected=sub_bi,
            full_set=new_V,
        )
        sub_result.derivation_steps = parent_steps + sub_result.derivation_steps
        return sub_result

    def _construct_hedge(
        self,
        graph: NDArray,
        bidirected: NDArray,
        treatment: Set[int],
        outcome: Set[int],
        V: Set[int],
    ) -> Tuple[Set[int], Set[int]]:
        """Construct a hedge (F, F') witnessing non-identifiability.

        A hedge for P(Y | do(X)) is a pair (F, F') where:
        - F' ⊂ F
        - F and F' are c-forests
        - F' contains Y and X ∩ F ⊆ F'
        """
        # The simplest hedge is (V, V\X) if V is a single c-component
        f_prime = V - treatment
        if not f_prime:
            f_prime = outcome.copy()
        return (V, f_prime)

    # =================================================================
    # C-component factorization
    # =================================================================

    def c_component_factorization(
        self,
        graph: NDArray[np.int_],
        bidirected: Optional[NDArray[np.int_]] = None,
    ) -> list[set[int]]:
        """Decompose the graph into confounded components (c-components).

        Two nodes belong to the same c-component if they are connected
        by a path consisting entirely of bidirected edges.  If no
        bidirected edges are given, each node is its own c-component.

        Parameters
        ----------
        graph : ndarray
            Directed adjacency matrix.
        bidirected : ndarray, optional
            Symmetric bidirected edge matrix.

        Returns
        -------
        list of set of int
        """
        graph = np.asarray(graph, dtype=np.float64)
        p = graph.shape[0]
        if bidirected is None:
            bidirected = np.zeros((p, p), dtype=np.float64)
        return self._find_c_components(graph, bidirected, set(range(p)))

    def _find_c_components(
        self,
        graph: NDArray,
        bidirected: NDArray,
        nodes: Set[int],
    ) -> List[Set[int]]:
        """Find c-components among *nodes*.

        Connected components in the bidirected-edge graph restricted to
        *nodes*.
        """
        if not nodes:
            return []

        idx_list = sorted(nodes)
        node_set = set(idx_list)
        visited: set[int] = set()
        components: list[set[int]] = []

        for start in idx_list:
            if start in visited:
                continue
            comp: set[int] = set()
            stack = [start]
            while stack:
                n = stack.pop()
                if n in visited:
                    continue
                visited.add(n)
                comp.add(n)
                # Traverse bidirected edges
                for nb in idx_list:
                    if nb != n and nb not in visited and nb in node_set:
                        if bidirected[n, nb] != 0 or bidirected[nb, n] != 0:
                            stack.append(nb)
            components.append(comp)
        return components

    # =================================================================
    # Find c-components (public alias)
    # =================================================================

    def find_c_components(
        self,
        graph: NDArray[np.int_],
        bidirected_edges: NDArray[np.int_],
    ) -> List[Set[int]]:
        """Find c-components (public wrapper).

        Parameters
        ----------
        graph : ndarray
        bidirected_edges : ndarray

        Returns
        -------
        list of set of int
        """
        graph = np.asarray(graph, dtype=np.float64)
        bidirected = np.asarray(bidirected_edges, dtype=np.float64)
        return self._find_c_components(
            graph, bidirected, set(range(graph.shape[0]))
        )

    # =================================================================
    # Hedge criterion
    # =================================================================

    def hedge_criterion(
        self,
        graph: NDArray[np.int_],
        treatment: set[int],
        outcome: set[int],
        *,
        bidirected: Optional[NDArray[np.int_]] = None,
    ) -> bool:
        """Check if the hedge criterion holds (effect NOT identifiable).

        Returns True if a hedge exists (non-identifiable), False
        otherwise.

        Parameters
        ----------
        graph : ndarray
        treatment, outcome : set of int
        bidirected : ndarray, optional

        Returns
        -------
        bool
        """
        hedge = self.find_hedge(graph, treatment, outcome, bidirected=bidirected)
        return hedge is not None

    def find_hedge(
        self,
        graph: NDArray[np.int_],
        treatment: set[int],
        outcome: set[int],
        *,
        bidirected: Optional[NDArray[np.int_]] = None,
    ) -> Optional[tuple[set[int], set[int]]]:
        """Find a hedge structure witnessing non-identifiability.

        A hedge for P(Y | do(X)) is a pair of R-rooted c-forests
        (F, F') with F' ⊂ F where F' contains a node in Y and a node
        in X ∩ F.

        Parameters
        ----------
        graph : ndarray
        treatment, outcome : set of int
        bidirected : ndarray, optional

        Returns
        -------
        (F, F') : tuple of set of int, or None
        """
        graph = np.asarray(graph, dtype=np.float64)
        p = graph.shape[0]
        if bidirected is None:
            bidirected = np.zeros((p, p), dtype=np.float64)
        else:
            bidirected = np.asarray(bidirected, dtype=np.float64)

        V = set(range(p))
        c_comps = self._find_c_components(graph, bidirected, V)

        # For each c-component containing a treatment node
        for comp in c_comps:
            if not (comp & treatment):
                continue
            # Check if the entire graph is a single c-component
            if comp == V:
                # V \ X must contain an outcome node
                f_prime = V - treatment
                if f_prime & outcome:
                    return (V, f_prime)

        # Check subsets: for each c-component S_i of G[V\X]
        remaining = V - treatment
        c_comps_rem = self._find_c_components(graph, bidirected, remaining)

        for s_i in c_comps_rem:
            # Check if S_i is a proper subset of a c-component of G
            for comp in c_comps:
                if s_i < comp and (comp & treatment) and (s_i & outcome):
                    return (comp, s_i)

        return None

    # =================================================================
    # Generalized adjustment criterion
    # =================================================================

    def generalized_adjustment(
        self,
        graph: NDArray[np.int_],
        treatment: set[int],
        outcome: set[int],
        *,
        bidirected: Optional[NDArray[np.int_]] = None,
    ) -> Optional[Set[int]]:
        """Generalized adjustment criterion (Perkovic et al., 2018).

        Finds a valid adjustment set Z such that Z satisfies:
        1. No node in Z is a descendant of any node on a proper causal
           path from X to Y (unless it is an ancestor of Y not through X).
        2. Z d-separates X from Y in the modified graph.

        Parameters
        ----------
        graph : ndarray
        treatment, outcome : set of int
        bidirected : ndarray, optional

        Returns
        -------
        set of int or None
            A valid adjustment set, or None if none exists.
        """
        graph = np.asarray(graph, dtype=np.float64)
        p = graph.shape[0]
        if bidirected is None:
            bidirected = np.zeros((p, p), dtype=np.float64)

        V = set(range(p))

        # Forbidden: descendants of treatment on proper causal paths
        desc_x = _descendants_of(graph, treatment)
        an_y = _ancestors_of(graph, outcome) | outcome
        forbidden = (treatment | outcome | (desc_x - an_y))
        candidates = sorted(V - forbidden)

        # Mutilated graph
        g_mut = graph.copy()
        for node in treatment:
            g_mut[:, node] = 0

        # Combined graph for d-separation (add bidirected as undirected)
        g_combined = g_mut.copy()

        # Try empty set
        if _d_separated(g_mut, treatment, outcome, set()):
            return set()

        # Try parents of X (common valid set)
        parents_x: set[int] = set()
        for x in treatment:
            parents_x |= set(_parents_of(graph, x))
        parents_x -= forbidden
        if parents_x and _d_separated(g_mut, treatment, outcome, parents_x):
            return parents_x

        # Try single variables
        for c in candidates:
            if _d_separated(g_mut, treatment, outcome, {c}):
                return {c}

        # Try pairs
        for i in range(len(candidates)):
            for j in range(i + 1, min(len(candidates), i + 10)):
                z = {candidates[i], candidates[j]}
                if _d_separated(g_mut, treatment, outcome, z):
                    return z

        # Try full candidate set
        full = set(candidates)
        if full and _d_separated(g_mut, treatment, outcome, full):
            return full

        return None

    # =================================================================
    # Find adjustment set
    # =================================================================

    def find_adjustment_set(
        self,
        graph: NDArray[np.int_],
        treatment: set[int],
        outcome: set[int],
        *,
        bidirected: Optional[NDArray[np.int_]] = None,
    ) -> Optional[set[int]]:
        """Find a valid adjustment set if one exists.

        First tries the standard back-door criterion, then falls back
        to the generalized adjustment criterion.

        Parameters
        ----------
        graph : ndarray
        treatment, outcome : set of int
        bidirected : ndarray, optional

        Returns
        -------
        set of int or None
        """
        graph = np.asarray(graph, dtype=np.float64)
        p = graph.shape[0]
        V = set(range(p))

        # Standard back-door: non-descendants of X
        desc_x = _descendants_of(graph, treatment)
        forbidden = treatment | outcome | desc_x
        candidates = sorted(V - forbidden)

        g_mut = graph.copy()
        for node in treatment:
            g_mut[:, node] = 0

        # Try empty set
        if _d_separated(g_mut, treatment, outcome, set()):
            return set()

        # Try parents of treatment
        parents_x: set[int] = set()
        for x in treatment:
            parents_x |= set(_parents_of(graph, x))
        parents_x -= forbidden
        if parents_x and _d_separated(g_mut, treatment, outcome, parents_x):
            return parents_x

        # Greedy search: start with all candidates and prune
        full = set(candidates)
        if not full:
            return self.generalized_adjustment(
                graph, treatment, outcome, bidirected=bidirected
            )

        if _d_separated(g_mut, treatment, outcome, full):
            # Try to minimize
            minimal = self._minimize_adjustment_set(
                g_mut, treatment, outcome, full
            )
            return minimal

        # Fall back to generalized adjustment
        return self.generalized_adjustment(
            graph, treatment, outcome, bidirected=bidirected
        )

    def _minimize_adjustment_set(
        self,
        g_mut: NDArray,
        treatment: Set[int],
        outcome: Set[int],
        valid_set: Set[int],
    ) -> Set[int]:
        """Greedily remove variables from a valid adjustment set."""
        current = set(valid_set)
        for v in sorted(valid_set):
            candidate = current - {v}
            if _d_separated(g_mut, treatment, outcome, candidate):
                current = candidate
        return current

    # =================================================================
    # Topological ID (recursive, district-based)
    # =================================================================

    def _topological_id(
        self,
        graph: NDArray,
        treatment: Set[int],
        outcome: Set[int],
        district: Set[int],
        *,
        bidirected: Optional[NDArray] = None,
    ) -> IdentifiabilityResult:
        """Recursive district-based identification.

        Uses the topological ordering to recursively decompose the
        identification problem.

        Parameters
        ----------
        graph : ndarray
        treatment, outcome, district : set of int
        bidirected : ndarray, optional

        Returns
        -------
        IdentifiabilityResult
        """
        p = graph.shape[0]
        if bidirected is None:
            bidirected = np.zeros((p, p), dtype=np.float64)

        V = set(range(p))
        steps: list[str] = []

        # Base case: single variable district
        if len(district) == 1:
            v = next(iter(district))
            pa_v = set(_parents_of(graph, v))
            if v in treatment:
                expr = f"P(x{v} | {_fmt(pa_v)})" if pa_v else f"P(x{v})"
            else:
                expr = f"P(x{v} | {_fmt(pa_v)})" if pa_v else f"P(x{v})"
            return IdentifiabilityResult(
                identifiable=True,
                estimand=expr,
                derivation_steps=[f"Base case: single node {v}"],
            )

        # Recurse on c-components of the district
        c_comps = self._find_c_components(graph, bidirected, district)

        if len(c_comps) == 1:
            # Single c-component — check for hedge
            an_y = _ancestors_of(graph, outcome) | outcome
            if district <= an_y and (district & treatment):
                return IdentifiabilityResult(
                    identifiable=False,
                    hedge=(district, district - treatment),
                    derivation_steps=[
                        f"Single c-component {_fmt(district)} "
                        f"with treatment — potential hedge"
                    ],
                )
            # Otherwise identifiable
            try:
                topo = _topological_sort(graph)
            except ValueError:
                topo = sorted(V)

            factors: list[str] = []
            for v in sorted(district):
                v_idx = topo.index(v) if v in topo else 0
                preds = set(topo[:v_idx]) & district
                if preds:
                    factors.append(f"P(x{v} | {_fmt(preds)})")
                else:
                    factors.append(f"P(x{v})")

            return IdentifiabilityResult(
                identifiable=True,
                estimand=" × ".join(factors),
                derivation_steps=[f"District {_fmt(district)} identified"],
            )

        # Multiple c-components: recurse
        parts: list[str] = []
        for comp in c_comps:
            sub = self._topological_id(
                graph, treatment, outcome, comp, bidirected=bidirected
            )
            if not sub.identifiable:
                return sub
            parts.append(sub.estimand or "?")

        return IdentifiabilityResult(
            identifiable=True,
            estimand=" × ".join(parts),
            derivation_steps=steps + [
                f"District decomposed into {len(c_comps)} sub-districts"
            ],
        )
