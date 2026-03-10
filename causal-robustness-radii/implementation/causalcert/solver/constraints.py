"""
Constraint generation for the ILP / LP formulations.

Provides modular constraint objects for:
1. **Acyclicity** — ensures the perturbed graph remains a DAG.
2. **CI consistency** — links edge-edit variables to d-separation predicates
   and CI test outcomes so that the solver only considers perturbations that
   genuinely alter the conclusion.
3. **Mutual exclusion** — prevents bidirectional edges.
4. **Budget** — limits total edit count.
5. **Back-door** — enforces or negates the back-door criterion.
6. **Conclusion negation** — no-good cuts excluding specific DAGs.
7. **Symmetry breaking** — orders equivalent nodes.
8. **Valid inequalities** — tightening cuts for LP relaxations.
9. **Lazy callbacks** — on-demand constraint generation.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from itertools import combinations
from typing import Any, Callable, Sequence

import numpy as np

from causalcert.types import (
    AdjacencyMatrix,
    CITestResult,
    EditType,
    NodeId,
    NodeSet,
    StructuralEdit,
)

logger = logging.getLogger(__name__)

_MIP_AVAILABLE = True
try:
    import mip as _mip
except ImportError:
    _MIP_AVAILABLE = False
    _mip = None  # type: ignore[assignment]


def _require_mip() -> None:
    """Raise a clear error when python-mip is not installed."""
    if not _MIP_AVAILABLE:
        raise ImportError(
            "python-mip is required for ILP/LP constraint generation. "
            "Install with: pip install mip"
        )


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class Constraint(ABC):
    """Abstract base for ILP/LP constraints."""

    @abstractmethod
    def add_to_model(self, model: Any, adj: AdjacencyMatrix) -> None:
        """Add this constraint family to a python-mip *model*.

        Parameters
        ----------
        model : Any
            python-mip Model.  Must have ``_x_vars`` (edge decision variables)
            already attached.
        adj : AdjacencyMatrix
            Original DAG adjacency matrix.
        """
        ...

    @property
    def name(self) -> str:
        """Human-readable constraint family name."""
        return self.__class__.__name__


# ---------------------------------------------------------------------------
# 1.  Acyclicity
# ---------------------------------------------------------------------------


class AcyclicityConstraint(Constraint):
    """Enforces acyclicity of the perturbed graph via topological-order variables.

    Uses auxiliary integer variables ``t[i]`` in {0, …, n-1} representing a
    topological ordering.  For every potential edge *i -> j*, the constraint
    ``t[j] >= t[i] + 1`` is enforced when the edge is present.

    Big-M encoding::

        t[j] - t[i] >= 1 - n * (1 - x[i,j])

    When ``x[i,j] = 1``:  ``t[j] >= t[i] + 1``  (ordering enforced).
    When ``x[i,j] = 0``:  ``t[j] >= t[i] + 1 - n``  (trivially satisfied).

    Parameters
    ----------
    n_nodes : int
        Number of nodes in the DAG.
    """

    def __init__(self, n_nodes: int) -> None:
        self.n_nodes = n_nodes

    # ----- helpers ----------------------------------------------------------

    def _ensure_topo_vars(self, model: Any) -> dict[int, Any]:
        """Create topological-order variables on *model* if absent."""
        _require_mip()
        if not hasattr(model, "_t_vars"):
            model._t_vars = {}
            n = self.n_nodes
            for i in range(n):
                model._t_vars[i] = model.add_var(
                    name=f"t_{i}",
                    var_type=_mip.INTEGER,
                    lb=0,
                    ub=n - 1,
                )
        return model._t_vars

    # ----- main entry -------------------------------------------------------

    def add_to_model(self, model: Any, adj: AdjacencyMatrix) -> None:
        _require_mip()
        n = self.n_nodes
        x = model._x_vars
        t = self._ensure_topo_vars(model)

        count = 0
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                model += (
                    t[j] - t[i] >= 1 - n * (1 - x[i, j]),
                    f"acyc_{i}_{j}",
                )
                count += 1

        logger.debug("Added %d acyclicity constraints for %d nodes", count, n)


# ---------------------------------------------------------------------------
# 2.  Mutual exclusion  (no bidirectional edges)
# ---------------------------------------------------------------------------


class MutualExclusionConstraint(Constraint):
    """Prevents bidirectional edges: ``x[i,j] + x[j,i] <= 1`` for i < j."""

    def __init__(self, n_nodes: int) -> None:
        self.n_nodes = n_nodes

    def add_to_model(self, model: Any, adj: AdjacencyMatrix) -> None:
        _require_mip()
        n = self.n_nodes
        x = model._x_vars
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                model += x[i, j] + x[j, i] <= 1, f"mutex_{i}_{j}"
                count += 1
        logger.debug("Added %d mutual-exclusion constraints", count)


# ---------------------------------------------------------------------------
# 3.  Budget
# ---------------------------------------------------------------------------


class BudgetConstraint(Constraint):
    """Limits the total number of structural edits to ``max_k``.

    Edit cost for each unordered pair {i,j} (i < j):

    * No original edge:   ``cost = x[i,j] + x[j,i]``   (0 or 1 add)
    * Original *i -> j*:  ``cost = 1 - x[i,j]``         (0 keep / 1 edit)
    * Original *j -> i*:  ``cost = 1 - x[j,i]``         (symmetric)

    Parameters
    ----------
    n_nodes : int
        Number of nodes.
    max_k : int
        Maximum allowed edit distance.
    """

    def __init__(self, n_nodes: int, max_k: int) -> None:
        self.n_nodes = n_nodes
        self.max_k = max_k

    def add_to_model(self, model: Any, adj: AdjacencyMatrix) -> None:
        _require_mip()
        cost_expr = build_edit_cost_expression(model, adj)
        model += cost_expr <= self.max_k, "budget"
        logger.debug("Added budget constraint: edits <= %d", self.max_k)


# ---------------------------------------------------------------------------
# 4.  CI consistency
# ---------------------------------------------------------------------------


class CIConsistencyConstraint(Constraint):
    """Links edge edits to conditional-independence implications.

    For each CI test result ``(x, y, S, reject)``:

    * **reject = True** (dependence):  at least one active path from *x*
      to *y* given *S* must be preserved in the target DAG.  Modelled with
      auxiliary binary *path-survival* variables.
    * **reject = False** (independence):  direct edges between *x* and *y*
      are forbidden (necessary condition for d-separation).

    Parameters
    ----------
    ci_results : Sequence[CITestResult]
        Pre-computed CI test results.
    treatment : NodeId
        Treatment variable.
    outcome : NodeId
        Outcome variable.
    """

    def __init__(
        self,
        ci_results: Sequence[CITestResult],
        treatment: NodeId,
        outcome: NodeId,
    ) -> None:
        self.ci_results = list(ci_results)
        self.treatment = treatment
        self.outcome = outcome

    def add_to_model(self, model: Any, adj: AdjacencyMatrix) -> None:
        _require_mip()
        n = adj.shape[0]
        x_vars = model._x_vars

        from causalcert.dag.dsep import DSeparationOracle

        oracle = DSeparationOracle(adj)
        count = 0

        for ci_idx, ci in enumerate(self.ci_results):
            cond = ci.conditioning_set
            xn, yn = ci.x, ci.y

            if ci.reject:
                # ----- dependence: preserve at least one active path -----
                active = oracle.active_paths(xn, yn, cond)
                if not active:
                    continue

                psurv: list[Any] = []
                for p_idx, path in enumerate(active[:8]):
                    evars: list[Any] = []
                    for k in range(len(path) - 1):
                        u, v = path[k], path[k + 1]
                        if 0 <= u < n and 0 <= v < n and u != v:
                            if adj[u, v]:
                                evars.append(x_vars[u, v])
                            elif adj[v, u]:
                                evars.append(x_vars[v, u])
                    if evars:
                        pv = model.add_var(
                            name=f"ps_{ci_idx}_{p_idx}",
                            var_type=_mip.BINARY,
                        )
                        for ev in evars:
                            model += pv <= ev
                        model += pv >= (
                            _mip.xsum(evars) - len(evars) + 1
                        )
                        psurv.append(pv)

                if psurv:
                    model += (
                        _mip.xsum(psurv) >= 1,
                        f"ci_dep_{xn}_{yn}_{ci_idx}",
                    )
                    count += 1
            else:
                # ----- independence: necessary blocking conditions -----
                if xn not in cond and yn not in cond:
                    model += (
                        x_vars[xn, yn] == 0,
                        f"ci_ind_f_{xn}_{yn}_{ci_idx}",
                    )
                    model += (
                        x_vars[yn, xn] == 0,
                        f"ci_ind_b_{xn}_{yn}_{ci_idx}",
                    )
                    count += 2

        logger.debug("Added %d CI-consistency constraints", count)


# ---------------------------------------------------------------------------
# 5.  Back-door criterion
# ---------------------------------------------------------------------------


class BackDoorConstraint(Constraint):
    """Enforces or negates the back-door criterion.

    When ``negate=False``: ensures a directed path from treatment to outcome
    exists (via network-flow encoding).  When ``negate=True``: ensures **no**
    directed path exists (the causal effect is unidentified).

    Parameters
    ----------
    treatment, outcome : NodeId
        Treatment and outcome node indices.
    negate : bool
        If ``True``, negate the criterion (forbid directed path).
    """

    def __init__(
        self,
        treatment: NodeId,
        outcome: NodeId,
        negate: bool = False,
    ) -> None:
        self.treatment = treatment
        self.outcome = outcome
        self.negate = negate

    def add_to_model(self, model: Any, adj: AdjacencyMatrix) -> None:
        _require_mip()
        n = adj.shape[0]
        x = model._x_vars
        t_nd = self.treatment
        y_nd = self.outcome

        # Network-flow encoding for directed reachability
        flow: dict[tuple[int, int], Any] = {}
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                flow[i, j] = model.add_var(
                    name=f"bdfl_{i}_{j}",
                    var_type=_mip.CONTINUOUS,
                    lb=0.0,
                    ub=1.0,
                )
                model += flow[i, j] <= x[i, j], f"bdfl_cap_{i}_{j}"

        # Flow conservation at intermediate nodes
        for v in range(n):
            if v == t_nd or v == y_nd:
                continue
            in_f = _mip.xsum(flow[u, v] for u in range(n) if u != v)
            out_f = _mip.xsum(flow[v, u] for u in range(n) if u != v)
            model += in_f == out_f, f"bdfl_cons_{v}"

        out_t = _mip.xsum(flow[t_nd, j] for j in range(n) if j != t_nd)
        in_y = _mip.xsum(flow[i, y_nd] for i in range(n) if i != y_nd)

        if self.negate:
            # No directed path: total flow == 0
            model += out_t == 0, "bd_neg_src"
            model += in_y == 0, "bd_neg_sink"
        else:
            # Directed path must exist: at least one unit of flow
            model += out_t >= 1, "bd_pos_src"
            model += in_y >= 1, "bd_pos_sink"

        logger.debug(
            "Added back-door %s constraints",
            "negation" if self.negate else "enforcement",
        )


# ---------------------------------------------------------------------------
# 6.  Conclusion-negation (no-good cuts)
# ---------------------------------------------------------------------------


class ConclusionNegationConstraint(Constraint):
    """No-good cut excluding a specific DAG from the feasible region.

    Adds: at least one edge variable must differ from *solution_adj*.

    Parameters
    ----------
    solution_adj : AdjacencyMatrix
        Adjacency matrix of the DAG to exclude.
    """

    _counter: int = 0  # class-level counter for unique names

    def __init__(self, solution_adj: AdjacencyMatrix) -> None:
        self.solution_adj = np.asarray(solution_adj, dtype=np.int8)
        ConclusionNegationConstraint._counter += 1
        self._id = ConclusionNegationConstraint._counter

    def add_to_model(self, model: Any, adj: AdjacencyMatrix) -> None:
        _require_mip()
        n = self.solution_adj.shape[0]
        x = model._x_vars

        diff_terms: list[Any] = []
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if self.solution_adj[i, j]:
                    diff_terms.append(1 - x[i, j])
                else:
                    diff_terms.append(x[i, j])

        if diff_terms:
            model += (
                _mip.xsum(diff_terms) >= 1,
                f"nogood_{self._id}",
            )


# ---------------------------------------------------------------------------
# 7.  Symmetry breaking
# ---------------------------------------------------------------------------


class SymmetryBreakingConstraint(Constraint):
    """Breaks symmetries among structurally equivalent nodes.

    Two nodes are *equivalent* if they have identical parent and child sets
    in the original DAG.  The constraint imposes ``t[v1] < t[v2]`` to
    select a canonical topological ordering among such nodes.

    Parameters
    ----------
    n_nodes : int
        Number of nodes in the DAG.
    """

    def __init__(self, n_nodes: int) -> None:
        self.n_nodes = n_nodes

    def add_to_model(self, model: Any, adj: AdjacencyMatrix) -> None:
        _require_mip()
        n = self.n_nodes
        if not hasattr(model, "_t_vars"):
            return
        t = model._t_vars

        groups: dict[tuple[frozenset[int], frozenset[int]], list[int]] = {}
        for v in range(n):
            parents = frozenset(int(p) for p in np.nonzero(adj[:, v])[0])
            children = frozenset(int(c) for c in np.nonzero(adj[v, :])[0])
            key = (parents, children)
            groups.setdefault(key, []).append(v)

        count = 0
        for nodes in groups.values():
            if len(nodes) <= 1:
                continue
            for k in range(len(nodes) - 1):
                model += (
                    t[nodes[k]] <= t[nodes[k + 1]] - 1,
                    f"sym_{nodes[k]}_{nodes[k+1]}",
                )
                count += 1

        logger.debug("Added %d symmetry-breaking constraints", count)


# ---------------------------------------------------------------------------
# 8.  Valid inequality tightener
# ---------------------------------------------------------------------------


class ValidInequalityTightener:
    """Generates families of valid inequalities to tighten LP relaxations.

    Cutting-plane families
    ----------------------
    1. **Triangle inequalities** — transitivity of the topological ordering.
    2. **Clique cuts** — from the moral graph structure.
    3. **Degree cuts** — bound vertex degrees.
    4. **Path cuts** — enforce reachability for causal paths.

    Parameters
    ----------
    adj : AdjacencyMatrix
        Original DAG adjacency matrix.
    """

    def __init__(self, adj: AdjacencyMatrix) -> None:
        self.adj = np.asarray(adj, dtype=np.int8)
        self.n = adj.shape[0]

    # ----- triangle --------------------------------------------------------

    def add_triangle_inequalities(self, model: Any, max_triples: int = 5000) -> int:
        """Add transitivity cuts: ``i->j`` and ``j->k`` imply ``t[k] >= t[i]+2``."""
        _require_mip()
        n = self.n
        x = model._x_vars
        t = model._t_vars
        count = 0

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                for k in range(n):
                    if k == i or k == j:
                        continue
                    model += (
                        t[k] - t[i] >= 2 - n * (2 - x[i, j] - x[j, k]),
                        f"tri_{i}_{j}_{k}",
                    )
                    count += 1
                    if count >= max_triples:
                        return count
        logger.debug("Added %d triangle inequalities", count)
        return count

    # ----- clique ----------------------------------------------------------

    def add_clique_cuts(self, model: Any) -> int:
        """Add clique cuts: edges within a moral-graph clique must form a DAG."""
        _require_mip()
        from causalcert.dag.moral import maximal_cliques, moral_graph

        mg = moral_graph(self.adj)
        cliques = maximal_cliques(mg)
        x = model._x_vars
        count = 0

        for cl in cliques:
            if len(cl) <= 2:
                continue
            nodes = sorted(cl)
            k = len(nodes)
            edge_sum: list[Any] = []
            for a_idx in range(k):
                for b_idx in range(a_idx + 1, k):
                    a, b = nodes[a_idx], nodes[b_idx]
                    edge_sum.append(x[a, b] + x[b, a])
            if edge_sum:
                model += (
                    _mip.xsum(edge_sum) <= k * (k - 1) // 2,
                    f"clq_{count}",
                )
                count += 1

        logger.debug("Added %d clique cuts", count)
        return count

    # ----- degree ----------------------------------------------------------

    def add_degree_cuts(self, model: Any, max_degree: int | None = None) -> int:
        """Limit maximum degree based on the original graph structure."""
        _require_mip()
        n = self.n
        x = model._x_vars
        if max_degree is None:
            orig_degrees = [
                int(self.adj[:, v].sum() + self.adj[v, :].sum())
                for v in range(n)
            ]
            max_degree = max(orig_degrees) + 3 if orig_degrees else n - 1
        count = 0
        for v in range(n):
            in_e = [x[u, v] for u in range(n) if u != v]
            out_e = [x[v, u] for u in range(n) if u != v]
            model += (
                _mip.xsum(in_e) + _mip.xsum(out_e) <= max_degree,
                f"deg_{v}",
            )
            count += 1
        logger.debug("Added %d degree cuts (max_degree=%d)", count, max_degree)
        return count

    # ----- path ------------------------------------------------------------

    def add_path_cuts(
        self, model: Any, src: NodeId, dst: NodeId, max_paths: int = 20
    ) -> int:
        """Add cuts enforcing reachability from *src* to *dst*."""
        _require_mip()
        from collections import deque

        n = self.n
        x = model._x_vars

        queue: deque[list[int]] = deque([[src]])
        visited: set[int] = {src}
        paths: list[list[int]] = []

        while queue and len(paths) < max_paths:
            path = queue.popleft()
            cur = path[-1]
            if cur == dst:
                paths.append(path)
                continue
            for j in range(n):
                if j not in visited:
                    visited.add(j)
                    queue.append(path + [j])

        count = 0
        for p in paths:
            evars = [x[p[k], p[k + 1]] for k in range(len(p) - 1)]
            if evars:
                model += _mip.xsum(evars) >= 1, f"pathcut_{count}"
                count += 1
        return count

    # ----- convenience -----------------------------------------------------

    def add_all(self, model: Any, *, treatment: int = -1, outcome: int = -1) -> int:
        """Add all tightening families.

        Skips expensive triangle cuts for large graphs (n > 15).
        """
        total = self.add_clique_cuts(model)
        total += self.add_degree_cuts(model)
        if self.n <= 15:
            total += self.add_triangle_inequalities(model)
        if treatment >= 0 and outcome >= 0:
            total += self.add_path_cuts(model, treatment, outcome)
        logger.debug("Total valid inequalities added: %d", total)
        return total


# ---------------------------------------------------------------------------
# 9.  Lazy constraint callback
# ---------------------------------------------------------------------------


class LazyConstraintCallback:
    """On-demand constraint generator for integer-solution callbacks.

    After the MIP solver finds an integer-feasible solution, this callback
    checks whether the conclusion predicate is overturned.  If not, a
    no-good cut is added to exclude the current solution.

    Attributes
    ----------
    n_cuts : int
        Number of no-good cuts added so far.
    n_checked : int
        Number of integer solutions evaluated.
    """

    def __init__(
        self,
        adj: AdjacencyMatrix,
        predicate: Callable[..., bool],
        data: Any,
        treatment: NodeId,
        outcome: NodeId,
    ) -> None:
        self.adj = np.asarray(adj, dtype=np.int8)
        self.predicate = predicate
        self.data = data
        self.treatment = treatment
        self.outcome = outcome
        self.n = adj.shape[0]
        self._cuts_added = 0
        self._solutions_checked = 0
        self._excluded_adjs: list[bytes] = []

    # ---- core method -------------------------------------------------------

    def check_and_cut(self, model: Any) -> bool:
        """Evaluate the current solution and add a no-good cut if needed.

        Returns
        -------
        bool
            ``True`` if the conclusion is **overturned** (witness found).
        """
        _require_mip()
        sol_adj = solution_adjacency(model, self.n)
        self._solutions_checked += 1

        pred_holds = self.predicate(
            sol_adj,
            self.data,
            treatment=self.treatment,
            outcome=self.outcome,
        )

        if pred_holds:
            key = sol_adj.tobytes()
            if key not in self._excluded_adjs:
                self._excluded_adjs.append(key)
                nogood = ConclusionNegationConstraint(sol_adj)
                nogood.add_to_model(model, self.adj)
                self._cuts_added += 1
            return False
        return True

    # ---- properties --------------------------------------------------------

    @property
    def n_cuts(self) -> int:
        return self._cuts_added

    @property
    def n_checked(self) -> int:
        return self._solutions_checked


# ---------------------------------------------------------------------------
# Module-level helper functions
# ---------------------------------------------------------------------------


def build_edit_cost_expression(model: Any, adj: AdjacencyMatrix) -> Any:
    """Build a linear expression for the total edit cost.

    For each unordered pair {i,j} with i < j the cost is computed as:

    * No original edge:  ``x[i,j] + x[j,i]``
    * Original *i->j*:   ``1 - x[i,j]``
    * Original *j->i*:   ``1 - x[j,i]``

    Parameters
    ----------
    model : mip.Model
        Model with ``_x_vars`` attribute.
    adj : AdjacencyMatrix
        Original DAG adjacency matrix.

    Returns
    -------
    mip.LinExpr
        Linear expression whose value equals the total edit count.
    """
    _require_mip()
    n = adj.shape[0]
    x = model._x_vars
    terms: list[Any] = []

    for i in range(n):
        for j in range(i + 1, n):
            aij = int(adj[i, j])
            aji = int(adj[j, i])
            if aij == 0 and aji == 0:
                terms.append(x[i, j])
                terms.append(x[j, i])
            elif aij == 1 and aji == 0:
                terms.append(1 - x[i, j])
            elif aij == 0 and aji == 1:
                terms.append(1 - x[j, i])
    return _mip.xsum(terms)


def extract_edits_from_model(
    model: Any,
    adj: AdjacencyMatrix,
) -> list[StructuralEdit]:
    """Extract structural edits from a solved model.

    Compares the original adjacency with the solution values of the edge
    decision variables to determine which edges changed.

    Parameters
    ----------
    model : mip.Model
        Solved model with ``_x_vars``.
    adj : AdjacencyMatrix
        Original DAG adjacency matrix.

    Returns
    -------
    list[StructuralEdit]
    """
    n = adj.shape[0]
    sol_adj = solution_adjacency(model, n)
    from causalcert.dag.edit import diff_edits

    return diff_edits(adj, sol_adj)


def solution_adjacency(model: Any, n: int) -> AdjacencyMatrix:
    """Extract the solution adjacency matrix from a solved MIP model.

    Parameters
    ----------
    model : mip.Model
        Solved model.
    n : int
        Number of nodes.

    Returns
    -------
    AdjacencyMatrix
    """
    x = model._x_vars
    sol = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            val = x[i, j].x
            if val is not None and val > 0.5:
                sol[i, j] = 1
    return sol


def compute_edit_cost(adj_orig: AdjacencyMatrix, adj_new: AdjacencyMatrix) -> int:
    """Compute the edit cost between two adjacency matrices.

    Accounts for reversals (each reversal counts as 1 edit, not 2).

    Parameters
    ----------
    adj_orig, adj_new : AdjacencyMatrix
        Original and modified adjacency matrices.

    Returns
    -------
    int
    """
    from causalcert.dag.edit import edit_distance

    return edit_distance(adj_orig, adj_new)
