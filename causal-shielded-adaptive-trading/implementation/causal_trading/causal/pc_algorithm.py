"""
PC and FCI algorithms for constraint-based causal structure learning.

Implements the PC algorithm (Spirtes, Glymour & Scheines 2000) with
skeleton discovery and Meek orientation rules, the stable PC variant
for order-independence, parallel conditional independence testing, and
the FCI extension for latent confounders (PAG output).
"""

from __future__ import annotations

import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Set,
    Sequence,
    Tuple,
    Union,
)

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from scipy import stats


# ====================================================================
# Default CI test: partial correlation + Fisher-z
# ====================================================================

def _partial_correlation(
    data: NDArray, i: int, j: int, cond: Sequence[int]
) -> float:
    """Compute partial correlation of columns *i* and *j* given *cond*
    using recursive residualisation (OLS)."""
    if len(cond) == 0:
        return float(np.corrcoef(data[:, i], data[:, j])[0, 1])

    cond = list(cond)
    Z = data[:, cond]
    # Residualise i and j on Z
    beta_i = np.linalg.lstsq(Z, data[:, i], rcond=None)[0]
    beta_j = np.linalg.lstsq(Z, data[:, j], rcond=None)[0]
    ri = data[:, i] - Z @ beta_i
    rj = data[:, j] - Z @ beta_j
    denom = np.sqrt(np.sum(ri ** 2) * np.sum(rj ** 2))
    if denom < 1e-15:
        return 0.0
    return float(np.dot(ri, rj) / denom)


def _fisher_z_test(
    data: NDArray,
    i: int,
    j: int,
    cond: Sequence[int],
    alpha: float = 0.05,
) -> Tuple[bool, float]:
    """Fisher-z test for conditional independence.

    Returns
    -------
    (independent, p_value)
    """
    n = data.shape[0]
    r = _partial_correlation(data, i, j, cond)
    r = np.clip(r, -1 + 1e-10, 1 - 1e-10)
    z = 0.5 * np.log((1 + r) / (1 - r))
    dof = max(n - len(cond) - 3, 1)
    test_stat = np.sqrt(dof) * abs(z)
    p_value = float(2 * (1 - stats.norm.cdf(test_stat)))
    return p_value > alpha, p_value


# ====================================================================
# Skeleton discovery
# ====================================================================

@dataclass
class SkeletonResult:
    """Result of the skeleton discovery phase."""
    adjacency: Dict[int, Set[int]]
    separation_sets: Dict[FrozenSet[int], Set[int]]
    max_cond_size: int = 0
    n_tests: int = 0


def _discover_skeleton(
    data: NDArray,
    ci_test: Callable,
    alpha: float,
    max_cond_size: Optional[int] = None,
    stable: bool = False,
    parallel: bool = False,
    n_workers: int = 4,
) -> SkeletonResult:
    """Skeleton discovery phase of the PC algorithm.

    Parameters
    ----------
    data : ndarray, shape (n, p)
    ci_test : callable(data, i, j, cond, alpha) -> (bool, float)
    alpha : float
        Significance level.
    max_cond_size : int or None
        Maximum conditioning set size (None = p-2).
    stable : bool
        Stable PC variant (do not remove adjacencies until end of each
        level).
    parallel : bool
        Run CI tests in parallel.
    n_workers : int
        Number of threads for parallel execution.
    """
    _, p = data.shape
    if max_cond_size is None:
        max_cond_size = p - 2

    adj: Dict[int, Set[int]] = {i: set(range(p)) - {i} for i in range(p)}
    sep_sets: Dict[FrozenSet[int], Set[int]] = {}
    n_tests = 0

    for cond_size in range(max_cond_size + 1):
        if stable:
            adj_snapshot = {i: set(s) for i, s in adj.items()}
        else:
            adj_snapshot = adj

        removals: List[Tuple[int, int, Set[int]]] = []
        test_queue: List[Tuple[int, int, Tuple[int, ...]]] = []

        for i in range(p):
            neighbors = sorted(adj_snapshot[i])
            for j in neighbors:
                if j <= i:
                    continue
                possible_cond = [k for k in neighbors if k != j]
                if len(possible_cond) < cond_size:
                    continue
                for cond in itertools.combinations(possible_cond, cond_size):
                    test_queue.append((i, j, cond))

        if not test_queue:
            if cond_size > 0:
                break
            continue

        if parallel and len(test_queue) > 10:
            results = _run_tests_parallel(
                data, ci_test, test_queue, alpha, n_workers
            )
        else:
            results = _run_tests_sequential(data, ci_test, test_queue, alpha)

        n_tests += len(results)

        for i, j, cond, independent, pval in results:
            if independent:
                removals.append((i, j, set(cond)))

        for i, j, cond_set in removals:
            adj[i].discard(j)
            adj[j].discard(i)
            sep_sets[frozenset({i, j})] = cond_set

        # Check if any node has enough neighbours to continue
        if all(len(adj[i]) < cond_size + 2 for i in range(p)):
            break

    return SkeletonResult(
        adjacency=adj,
        separation_sets=sep_sets,
        max_cond_size=cond_size,
        n_tests=n_tests,
    )


def _run_tests_sequential(
    data, ci_test, test_queue, alpha
) -> List[Tuple[int, int, Tuple, bool, float]]:
    results = []
    for i, j, cond in test_queue:
        independent, pval = ci_test(data, i, j, cond, alpha)
        results.append((i, j, cond, independent, pval))
    return results


def _run_tests_parallel(
    data, ci_test, test_queue, alpha, n_workers
) -> List[Tuple[int, int, Tuple, bool, float]]:
    results = []
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {}
        for i, j, cond in test_queue:
            f = executor.submit(ci_test, data, i, j, cond, alpha)
            futures[f] = (i, j, cond)
        for f in as_completed(futures):
            i, j, cond = futures[f]
            independent, pval = f.result()
            results.append((i, j, cond, independent, pval))
    return results


# ====================================================================
# Orientation rules (Meek 1995)
# ====================================================================

class _OrientationHelper:
    """Mutable edge state for orientation rules.

    Edge marks: '-' (tail), '>' (arrowhead), 'o' (circle / unknown).
    Stored as mark[i][j] = mark at the j-end of the i—j edge.
    """

    def __init__(
        self,
        adj: Dict[int, Set[int]],
        sep_sets: Dict[FrozenSet[int], Set[int]],
    ) -> None:
        self.adj = {i: set(s) for i, s in adj.items()}
        self.sep_sets = dict(sep_sets)
        self.mark: Dict[int, Dict[int, str]] = {}
        for i, nbrs in self.adj.items():
            self.mark[i] = {}
            for j in nbrs:
                self.mark[i][j] = "-"

    def orient(self, i: int, j: int) -> None:
        """Orient edge as i → j (tail at i, head at j)."""
        self.mark[i][j] = ">"
        self.mark[j][i] = "-"

    def is_oriented(self, i: int, j: int) -> bool:
        return self.mark[i].get(j) == ">"

    def is_undirected(self, i: int, j: int) -> bool:
        return (
            self.mark[i].get(j) == "-"
            and self.mark[j].get(i) == "-"
        )

    def is_adjacent(self, i: int, j: int) -> bool:
        return j in self.adj.get(i, set())

    # ----- V-structure orientation (Step 1)
    def orient_v_structures(self) -> int:
        """Orient v-structures: for each triple i - k - j where i and j
        are non-adjacent, orient i → k ← j iff k ∉ sep(i, j)."""
        count = 0
        nodes = sorted(self.adj.keys())
        for k in nodes:
            nbrs = sorted(self.adj[k])
            for idx_a, i in enumerate(nbrs):
                for j in nbrs[idx_a + 1:]:
                    if self.is_adjacent(i, j):
                        continue
                    sep = self.sep_sets.get(frozenset({i, j}), set())
                    if k not in sep:
                        self.orient(i, k)
                        self.orient(j, k)
                        count += 1
        return count

    # ----- Meek rules (Step 2)
    def apply_meek_rules(self, max_iter: int = 100) -> int:
        """Iteratively apply Meek's four orientation rules."""
        total = 0
        for _ in range(max_iter):
            changed = 0
            changed += self._meek_r1()
            changed += self._meek_r2()
            changed += self._meek_r3()
            changed += self._meek_r4()
            total += changed
            if changed == 0:
                break
        return total

    def _meek_r1(self) -> int:
        """R1: i → k - j  ⇒  k → j  (if i and j non-adjacent)."""
        count = 0
        for k in sorted(self.adj):
            for j in sorted(self.adj[k]):
                if not self.is_undirected(k, j):
                    continue
                for i in self.adj[k]:
                    if i == j:
                        continue
                    if self.is_oriented(i, k) and not self.is_adjacent(i, j):
                        self.orient(k, j)
                        count += 1
                        break
        return count

    def _meek_r2(self) -> int:
        """R2: i → k → j and i - j  ⇒  i → j."""
        count = 0
        for i in sorted(self.adj):
            for j in sorted(self.adj[i]):
                if not self.is_undirected(i, j):
                    continue
                for k in self.adj[i]:
                    if k == j:
                        continue
                    if (
                        self.is_oriented(i, k)
                        and self.is_adjacent(k, j)
                        and self.is_oriented(k, j)
                    ):
                        self.orient(i, j)
                        count += 1
                        break
        return count

    def _meek_r3(self) -> int:
        """R3: i - k, i - j, k → l, j → l, i - l  ⇒  i → l."""
        count = 0
        for i in sorted(self.adj):
            for l_ in sorted(self.adj[i]):
                if not self.is_undirected(i, l_):
                    continue
                nbrs_i = self.adj[i]
                for k in nbrs_i:
                    if k == l_:
                        continue
                    if not self.is_undirected(i, k):
                        continue
                    if not (self.is_adjacent(k, l_) and self.is_oriented(k, l_)):
                        continue
                    for j in nbrs_i:
                        if j == l_ or j == k:
                            continue
                        if not self.is_undirected(i, j):
                            continue
                        if self.is_adjacent(k, j):
                            continue
                        if self.is_adjacent(j, l_) and self.is_oriented(j, l_):
                            self.orient(i, l_)
                            count += 1
                            break
                    else:
                        continue
                    break
        return count

    def _meek_r4(self) -> int:
        """R4: i - k, k → l → j, i - j  ⇒  i → j  (if i adj l)."""
        count = 0
        for i in sorted(self.adj):
            for j in sorted(self.adj[i]):
                if not self.is_undirected(i, j):
                    continue
                for k in self.adj[i]:
                    if k == j:
                        continue
                    if not self.is_undirected(i, k):
                        continue
                    if not self.is_adjacent(k, j):
                        continue
                    for l_ in self.adj[k]:
                        if l_ == i or l_ == j:
                            continue
                        if not self.is_oriented(k, l_):
                            continue
                        if (
                            self.is_adjacent(l_, j)
                            and self.is_oriented(l_, j)
                            and self.is_adjacent(i, l_)
                        ):
                            self.orient(i, j)
                            count += 1
                            break
                    else:
                        continue
                    break
        return count

    def to_digraph(self, node_names: Optional[List[str]] = None) -> nx.DiGraph:
        G = nx.DiGraph()
        names = node_names or [str(i) for i in sorted(self.adj)]
        G.add_nodes_from(names)
        for i in self.adj:
            for j in self.adj[i]:
                if self.is_oriented(i, j):
                    G.add_edge(names[i], names[j])
        return G

    def to_undirected(self, node_names: Optional[List[str]] = None) -> nx.Graph:
        G = nx.Graph()
        names = node_names or [str(i) for i in sorted(self.adj)]
        G.add_nodes_from(names)
        for i in self.adj:
            for j in self.adj[i]:
                if j > i:
                    G.add_edge(names[i], names[j])
        return G


# ====================================================================
# PC Algorithm
# ====================================================================

class PCAlgorithm:
    """Constraint-based causal discovery using the PC algorithm.

    Parameters
    ----------
    alpha : float
        Significance level for conditional independence tests.
    ci_test : callable or None
        Custom CI test function ``(data, i, j, cond, alpha) -> (bool, pval)``.
        Defaults to Fisher-z partial-correlation test.
    max_cond_size : int or None
        Maximum conditioning set size (None = no limit).
    stable : bool
        Use the stable PC variant (Colombo & Maathuis 2014).
    parallel : bool
        Run CI tests in parallel.
    n_workers : int
        Thread count for parallel mode.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        ci_test: Optional[Callable] = None,
        max_cond_size: Optional[int] = None,
        stable: bool = True,
        parallel: bool = False,
        n_workers: int = 4,
    ) -> None:
        self.alpha = alpha
        self.ci_test = ci_test or _fisher_z_test
        self.max_cond_size = max_cond_size
        self.stable = stable
        self.parallel = parallel
        self.n_workers = n_workers

        self._skeleton: Optional[SkeletonResult] = None
        self._orient: Optional[_OrientationHelper] = None
        self._variable_names: Optional[List[str]] = None

    def fit(
        self,
        data: NDArray,
        variable_names: Optional[List[str]] = None,
    ) -> "PCAlgorithm":
        """Run the PC algorithm on *data*.

        Parameters
        ----------
        data : ndarray, shape (n, p)
        variable_names : list of str, optional
        """
        data = np.asarray(data, dtype=np.float64)
        _, p = data.shape
        self._variable_names = variable_names if variable_names is not None else list(range(p))

        # Phase 1: skeleton
        self._skeleton = _discover_skeleton(
            data,
            ci_test=self.ci_test,
            alpha=self.alpha,
            max_cond_size=self.max_cond_size,
            stable=self.stable,
            parallel=self.parallel,
            n_workers=self.n_workers,
        )

        # Phase 2: orientation
        self._orient = _OrientationHelper(
            self._skeleton.adjacency, self._skeleton.separation_sets
        )
        self._orient.orient_v_structures()
        self._orient.apply_meek_rules()
        return self

    def get_skeleton(self) -> nx.Graph:
        """Return the undirected skeleton graph."""
        if self._orient is None:
            raise RuntimeError("Call fit() first.")
        return self._orient.to_undirected(self._variable_names)

    def get_dag(self) -> nx.DiGraph:
        """Return the (partially) oriented DAG."""
        if self._orient is None:
            raise RuntimeError("Call fit() first.")
        return self._orient.to_digraph(self._variable_names)

    def get_cpdag(self) -> nx.DiGraph:
        """Return the CPDAG (completed partially directed acyclic graph).

        Directed edges represent compelled orientations; undirected edges
        (represented as two directed edges) represent reversible edges.
        """
        if self._orient is None:
            raise RuntimeError("Call fit() first.")
        names = self._variable_names
        G = nx.DiGraph()
        G.add_nodes_from(names)
        for i in self._orient.adj:
            for j in self._orient.adj[i]:
                if j <= i:
                    continue
                ni, nj = names[i], names[j]
                if self._orient.is_oriented(i, j):
                    G.add_edge(ni, nj, compelled=True)
                elif self._orient.is_oriented(j, i):
                    G.add_edge(nj, ni, compelled=True)
                else:
                    G.add_edge(ni, nj, compelled=False)
                    G.add_edge(nj, ni, compelled=False)
        return G

    @property
    def separation_sets(self) -> Dict[FrozenSet[str], Set[str]]:
        if self._skeleton is None:
            raise RuntimeError("Call fit() first.")
        names = self._variable_names
        out: Dict[FrozenSet[str], Set[str]] = {}
        for key, val in self._skeleton.separation_sets.items():
            str_key = frozenset(names[i] for i in key)
            str_val = {names[i] for i in val}
            out[str_key] = str_val
        return out


# ====================================================================
# Stable PC (convenience subclass)
# ====================================================================

class StablePCAlgorithm(PCAlgorithm):
    """Order-independent (stable) variant of the PC algorithm."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs["stable"] = True
        super().__init__(**kwargs)


# ====================================================================
# FCI Algorithm (extension for latent confounders)
# ====================================================================

class FCIAlgorithm:
    """Fast Causal Inference algorithm (Spirtes et al. 2000).

    Extends PC to handle latent confounders, producing a Partial Ancestral
    Graph (PAG) with circle marks for ambiguous edge endpoints.

    Parameters
    ----------
    alpha : float
        Significance level.
    ci_test : callable or None
        CI test function.
    max_cond_size : int or None
        Max conditioning set size.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        ci_test: Optional[Callable] = None,
        max_cond_size: Optional[int] = None,
    ) -> None:
        self.alpha = alpha
        self.ci_test = ci_test or _fisher_z_test
        self.max_cond_size = max_cond_size

        self._variable_names: Optional[List[str]] = None
        self._skeleton: Optional[SkeletonResult] = None
        self._pag_marks: Optional[Dict[int, Dict[int, str]]] = None

    def fit(
        self,
        data: NDArray,
        variable_names: Optional[List[str]] = None,
    ) -> "FCIAlgorithm":
        data = np.asarray(data, dtype=np.float64)
        _, p = data.shape
        self._variable_names = variable_names or [f"X{i}" for i in range(p)]

        # Step 1: skeleton (same as PC)
        self._skeleton = _discover_skeleton(
            data,
            ci_test=self.ci_test,
            alpha=self.alpha,
            max_cond_size=self.max_cond_size,
            stable=True,
        )
        adj = self._skeleton.adjacency
        sep_sets = self._skeleton.separation_sets

        # Initialise edge marks to 'o' (circle)
        self._pag_marks = {}
        for i in adj:
            self._pag_marks[i] = {}
            for j in adj[i]:
                self._pag_marks[i][j] = "o"

        # Step 2: orient definite colliders (as in PC)
        for k in sorted(adj):
            nbrs = sorted(adj[k])
            for ia, i in enumerate(nbrs):
                for j in nbrs[ia + 1:]:
                    if j in adj.get(i, set()):
                        continue
                    sep = sep_sets.get(frozenset({i, j}), set())
                    if k not in sep:
                        self._pag_marks[i][k] = ">"
                        self._pag_marks[j][k] = ">"

        # Step 3: FCI orientation rules (R1-R4 + R8-R10)
        self._apply_fci_rules(adj, sep_sets, data)
        return self

    def _apply_fci_rules(
        self,
        adj: Dict[int, Set[int]],
        sep_sets: Dict[FrozenSet[int], Set[int]],
        data: NDArray,
        max_iter: int = 100,
    ) -> None:
        marks = self._pag_marks
        for _ in range(max_iter):
            changed = 0
            changed += self._fci_r1(adj, marks)
            changed += self._fci_r2(adj, marks)
            changed += self._fci_r3(adj, marks)
            changed += self._fci_r4(adj, marks, sep_sets)
            changed += self._fci_r8(adj, marks)
            changed += self._fci_r9(adj, marks)
            changed += self._fci_r10(adj, marks)
            if changed == 0:
                break

    def _fci_r1(self, adj, marks) -> int:
        """R1: α *→ β o-* γ, α not adj γ  ⇒  β → γ (orient as non-collider)."""
        count = 0
        for b in sorted(adj):
            for g in sorted(adj[b]):
                if marks[b].get(g) != "o":
                    continue
                for a in adj[b]:
                    if a == g:
                        continue
                    if marks[a].get(b) == ">" and g not in adj.get(a, set()):
                        marks[b][g] = ">"
                        marks[g][b] = "-"
                        count += 1
                        break
        return count

    def _fci_r2(self, adj, marks) -> int:
        """R2: α → β *→ γ  or  α *→ β → γ, and α *-o γ  ⇒  α *→ γ."""
        count = 0
        for a in sorted(adj):
            for g in sorted(adj[a]):
                if marks[a].get(g) != "o" or g not in adj[a]:
                    continue
                for b in adj[a]:
                    if b == g or b not in adj.get(g, set()):
                        continue
                    cond1 = marks[a].get(b) == ">" and marks[b].get(g) == ">"
                    cond2 = marks[b].get(a) == "-" and marks[b].get(g) == ">"
                    if cond1 or cond2:
                        marks[a][g] = ">"
                        count += 1
                        break
        return count

    def _fci_r3(self, adj, marks) -> int:
        """R3: α *→ β ←* γ and α *-o θ o-* γ, θ *-o β,
        α not adj γ  ⇒  θ *→ β."""
        count = 0
        for b in sorted(adj):
            nbrs = sorted(adj[b])
            for ia, a in enumerate(nbrs):
                if marks[a].get(b) != ">":
                    continue
                for g in nbrs[ia + 1:]:
                    if marks[g].get(b) != ">":
                        continue
                    if g in adj.get(a, set()):
                        continue
                    for t in adj[b]:
                        if t == a or t == g:
                            continue
                        if marks[b].get(t) != "o":
                            continue
                        if t in adj.get(a, set()) and t in adj.get(g, set()):
                            if marks[a].get(t) == "o" and marks[g].get(t) == "o":
                                marks[t][b] = ">"
                                count += 1
        return count

    def _fci_r4(self, adj, marks, sep_sets) -> int:
        """R4: discriminating path rule."""
        # Simplified: skip complex discriminating path search
        return 0

    def _fci_r8(self, adj, marks) -> int:
        """R8: α → β → γ  and  α o→ γ  ⇒  α → γ."""
        count = 0
        for a in sorted(adj):
            for g in sorted(adj[a]):
                if marks[a].get(g) != ">" or marks[g].get(a) != "o":
                    continue
                for b in adj[a]:
                    if b == g:
                        continue
                    if (
                        marks[a].get(b) == ">"
                        and marks[b].get(a) == "-"
                        and b in adj.get(g, set())
                        and marks[b].get(g) == ">"
                        and marks[g].get(b) == "-"
                    ):
                        marks[g][a] = "-"
                        count += 1
                        break
        return count

    def _fci_r9(self, adj, marks) -> int:
        """R9: α o→ γ  and  α — ... — γ (undirected path of circle marks)
        ⇒  α → γ.  Simplified: if α o→ γ and there exists β s.t.
        α o-o β and β → γ and β not adj to … (one-step)."""
        count = 0
        for a in sorted(adj):
            for g in sorted(adj[a]):
                if marks[a].get(g) != ">" or marks[g].get(a) != "o":
                    continue
                # Look for uncovered potentially directed path from α to γ
                for b in adj[a]:
                    if b == g:
                        continue
                    if b not in adj.get(g, set()):
                        continue
                    if (
                        marks[a].get(b) in ("o", "-")
                        and marks[b].get(g) == ">"
                        and marks[g].get(b) == "-"
                    ):
                        marks[g][a] = "-"
                        count += 1
                        break
        return count

    def _fci_r10(self, adj, marks) -> int:
        """R10: α o→ γ and α → β → γ and α → δ → γ and β o→ δ
        ⇒  α → γ."""
        count = 0
        for a in sorted(adj):
            for g in sorted(adj[a]):
                if marks[a].get(g) != ">" or marks[g].get(a) != "o":
                    continue
                directed_parents = [
                    b for b in adj.get(g, set())
                    if b != a
                    and marks[b].get(g) == ">"
                    and marks[g].get(b) == "-"
                    and b in adj.get(a, set())
                    and marks[a].get(b) == ">"
                    and marks[b].get(a) == "-"
                ]
                if len(directed_parents) >= 2:
                    # Check if any pair has a circle-arrow
                    for ib, b in enumerate(directed_parents):
                        for d in directed_parents[ib + 1:]:
                            if b in adj.get(d, set()):
                                if marks[b].get(d) == ">" and marks[d].get(b) == "o":
                                    marks[g][a] = "-"
                                    count += 1
                                    break
                        else:
                            continue
                        break
        return count

    def get_pag(self) -> nx.DiGraph:
        """Return the PAG as a DiGraph with edge-mark attributes."""
        if self._pag_marks is None:
            raise RuntimeError("Call fit() first.")
        names = self._variable_names
        G = nx.DiGraph()
        G.add_nodes_from(names)
        adj = self._skeleton.adjacency
        for i in adj:
            for j in adj[i]:
                if j <= i:
                    continue
                ni, nj = names[i], names[j]
                mi = self._pag_marks[i][j]  # mark at j-end
                mj = self._pag_marks[j][i]  # mark at i-end
                G.add_edge(ni, nj, mark_at_target=mi, mark_at_source=mj)
                G.add_edge(nj, ni, mark_at_target=mj, mark_at_source=mi)
        return G

    def get_skeleton(self) -> nx.Graph:
        if self._skeleton is None:
            raise RuntimeError("Call fit() first.")
        names = self._variable_names
        G = nx.Graph()
        G.add_nodes_from(names)
        for i in self._skeleton.adjacency:
            for j in self._skeleton.adjacency[i]:
                if j > i:
                    G.add_edge(names[i], names[j])
        return G


# ====================================================================
# Alpha-adaptive CI testing
# ====================================================================

def alpha_adaptive_test(
    data: NDArray,
    i: int,
    j: int,
    cond: Sequence[int],
    alpha_base: float = 0.05,
) -> Tuple[bool, float]:
    """Adaptive significance level that decreases with conditioning set
    size to control false discovery rate:  α_eff = α_base / (|cond| + 1).
    """
    alpha_eff = alpha_base / (len(cond) + 1)
    return _fisher_z_test(data, i, j, cond, alpha_eff)
