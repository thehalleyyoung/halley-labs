"""Independent per-context + post-hoc comparison baseline (BL1).

Runs a constraint-based structure learner (PC algorithm) independently on
each context dataset, then compares the resulting DAGs post-hoc to classify
edge plasticity.  This is the naïve baseline: no information is shared
across contexts during learning.

References
----------
Spirtes, Glymour & Scheines (2000).  *Causation, Prediction, and Search*.
"""

from __future__ import annotations

import itertools
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from cpa.core.types import PlasticityClass


# -------------------------------------------------------------------
# Helpers – simple PC algorithm (Fisher-z CI tests)
# -------------------------------------------------------------------


def _partial_correlation(
    data: NDArray, i: int, j: int, cond: List[int],
) -> float:
    """Compute partial correlation of *i* and *j* given *cond* via regression."""
    if len(cond) == 0:
        return float(np.corrcoef(data[:, i], data[:, j])[0, 1])
    Z = data[:, cond]
    # Regress i and j on cond
    Z_aug = np.column_stack([Z, np.ones(Z.shape[0])])
    beta_i, _, _, _ = np.linalg.lstsq(Z_aug, data[:, i], rcond=None)
    beta_j, _, _, _ = np.linalg.lstsq(Z_aug, data[:, j], rcond=None)
    res_i = data[:, i] - Z_aug @ beta_i
    res_j = data[:, j] - Z_aug @ beta_j
    denom = np.sqrt(np.sum(res_i ** 2) * np.sum(res_j ** 2))
    if denom < 1e-15:
        return 0.0
    return float(np.sum(res_i * res_j) / denom)


def _fisher_z_test(
    data: NDArray, i: int, j: int, cond: List[int], alpha: float,
) -> Tuple[bool, float]:
    """Fisher-z test for conditional independence.

    Returns (independent, p_value).
    """
    n = data.shape[0]
    r = _partial_correlation(data, i, j, cond)
    r = np.clip(r, -0.9999, 0.9999)
    z = 0.5 * np.log((1.0 + r) / (1.0 - r))
    se = 1.0 / np.sqrt(max(n - len(cond) - 3, 1))
    stat = abs(z) / se
    p_value = float(2.0 * (1.0 - stats.norm.cdf(stat)))
    return p_value > alpha, p_value


def _pc_skeleton(
    data: NDArray, alpha: float, max_cond: Optional[int] = None,
) -> Tuple[NDArray, Dict[Tuple[int, int], List[int]]]:
    """Learn the PC skeleton (undirected adjacency + separating sets).

    Parameters
    ----------
    data : NDArray, shape (n, p)
    alpha : significance level
    max_cond : maximum conditioning-set size (None = p-2)

    Returns
    -------
    adj : NDArray, shape (p, p)  – symmetric binary adjacency
    sep_sets : mapping (i, j) -> conditioning set that separated them
    """
    p = data.shape[1]
    adj = np.ones((p, p), dtype=np.float64) - np.eye(p)
    sep_sets: Dict[Tuple[int, int], List[int]] = {}
    if max_cond is None:
        max_cond = p - 2

    for depth in range(max_cond + 1):
        for i in range(p):
            for j in range(i + 1, p):
                if adj[i, j] == 0:
                    continue
                neighbours_i = [
                    k for k in range(p) if k != j and adj[i, k] != 0
                ]
                if len(neighbours_i) < depth:
                    continue
                found_sep = False
                for cond in itertools.combinations(neighbours_i, depth):
                    cond_list = list(cond)
                    indep, _ = _fisher_z_test(data, i, j, cond_list, alpha)
                    if indep:
                        adj[i, j] = adj[j, i] = 0.0
                        sep_sets[(i, j)] = cond_list
                        sep_sets[(j, i)] = cond_list
                        found_sep = True
                        break
                if found_sep:
                    continue
                # Also try conditioning sets from neighbours of j
                neighbours_j = [
                    k for k in range(p) if k != i and adj[j, k] != 0
                ]
                if len(neighbours_j) >= depth:
                    for cond in itertools.combinations(neighbours_j, depth):
                        cond_list = list(cond)
                        indep, _ = _fisher_z_test(
                            data, i, j, cond_list, alpha,
                        )
                        if indep:
                            adj[i, j] = adj[j, i] = 0.0
                            sep_sets[(i, j)] = cond_list
                            sep_sets[(j, i)] = cond_list
                            break
    return adj, sep_sets


def _orient_v_structures(
    adj: NDArray, sep_sets: Dict[Tuple[int, int], List[int]],
) -> NDArray:
    """Orient v-structures: i -> k <- j when k not in sep(i, j)."""
    p = adj.shape[0]
    dag = adj.copy()
    for k in range(p):
        parents = [i for i in range(p) if adj[i, k] != 0 and i != k]
        for a, b in itertools.combinations(parents, 2):
            if adj[a, b] != 0 or adj[b, a] != 0:
                continue  # a and b are adjacent, skip
            sep = sep_sets.get((a, b), sep_sets.get((b, a), []))
            if k not in sep:
                # Orient a -> k <- b
                dag[k, a] = 0.0
                dag[k, b] = 0.0
    return dag


def _apply_meek_rules(dag: NDArray) -> NDArray:
    """Apply Meek's orientation rules R1-R3 until convergence."""
    p = dag.shape[0]
    changed = True
    while changed:
        changed = False
        for i in range(p):
            for j in range(p):
                if i == j:
                    continue
                if dag[i, j] == 0 or dag[j, i] == 0:
                    continue  # already directed or absent
                # R1: i -> k - j  =>  i -> k -> j (if i and j not adjacent
                #     through some other path, we orient k -> j)
                for k in range(p):
                    if k == i or k == j:
                        continue
                    # i -> k (directed) and k - j (undirected)
                    if (dag[i, k] != 0 and dag[k, i] == 0
                            and dag[k, j] != 0 and dag[j, k] != 0
                            and dag[i, j] == 0):
                        dag[j, k] = 0.0
                        changed = True
                # R2: i - j and i -> k -> j  =>  i -> j
                for k in range(p):
                    if k == i or k == j:
                        continue
                    if (dag[i, k] != 0 and dag[k, i] == 0
                            and dag[k, j] != 0 and dag[j, k] == 0
                            and dag[i, j] != 0 and dag[j, i] != 0):
                        dag[j, i] = 0.0
                        changed = True
    return dag


def _pc_algorithm(
    data: NDArray, alpha: float = 0.05, max_cond: Optional[int] = None,
) -> NDArray:
    """Run the full PC algorithm and return a CPDAG adjacency matrix.

    Entry ``adj[i, j] != 0`` means i -> j (or i - j if both directions).
    """
    skeleton, sep_sets = _pc_skeleton(data, alpha, max_cond)
    cpdag = _orient_v_structures(skeleton, sep_sets)
    cpdag = _apply_meek_rules(cpdag)
    return cpdag


# -------------------------------------------------------------------
# Structural Hamming Distance
# -------------------------------------------------------------------


def _structural_hamming_distance(dag1: NDArray, dag2: NDArray) -> int:
    """Compute the Structural Hamming Distance between two (CP)DAGs.

    Counts differences in edge presence/orientation.
    """
    p = dag1.shape[0]
    shd = 0
    for i in range(p):
        for j in range(i + 1, p):
            e1_ij, e1_ji = dag1[i, j] != 0, dag1[j, i] != 0
            e2_ij, e2_ji = dag2[i, j] != 0, dag2[j, i] != 0
            if (e1_ij, e1_ji) != (e2_ij, e2_ji):
                shd += 1
    return shd


# -------------------------------------------------------------------
# Edge-level comparison utilities
# -------------------------------------------------------------------


def _collect_edges(dag: NDArray) -> Set[Tuple[int, int]]:
    """Return the set of directed edges in *dag*."""
    edges: Set[Tuple[int, int]] = set()
    p = dag.shape[0]
    for i in range(p):
        for j in range(p):
            if i != j and dag[i, j] != 0:
                edges.add((i, j))
    return edges


def _edge_presence_matrix(
    dags: Dict[str, NDArray], p: int,
) -> NDArray:
    """Return a (p, p, n_contexts) binary tensor of edge presence."""
    ctx_keys = sorted(dags.keys())
    n_ctx = len(ctx_keys)
    tensor = np.zeros((p, p, n_ctx), dtype=np.float64)
    for k, key in enumerate(ctx_keys):
        dag = dags[key]
        tensor[:, :, k] = (dag != 0).astype(np.float64)
    return tensor


# -------------------------------------------------------------------
# Main class
# -------------------------------------------------------------------


class IndependentPHC:
    """Independent per-context learning with post-hoc comparison (BL1).

    Learns a CPDAG independently per context via the PC algorithm, then
    compares DAGs post-hoc to classify edge plasticity.

    Parameters
    ----------
    learner : str
        Structure learning algorithm (currently ``"pc"``).
    significance_level : float
        Alpha for conditional independence tests.
    comparison_method : str
        Metric for comparing DAGs (``"shd"`` or ``"edge"``).
    """

    def __init__(
        self,
        learner: str = "pc",
        significance_level: float = 0.05,
        comparison_method: str = "shd",
    ) -> None:
        if learner not in ("pc",):
            raise ValueError(f"Unsupported learner: {learner!r}")
        self._learner = learner
        self._alpha = significance_level
        self._comparison_method = comparison_method
        self._dags: Dict[str, NDArray] = {}
        self._results: Dict[str, Any] = {}
        self._plasticity: Dict[Tuple[int, int], PlasticityClass] = {}
        self._n_vars: int = 0
        self._fitted: bool = False

    # ---------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------

    def fit(
        self,
        datasets: Dict[str, NDArray],
        context_labels: Optional[List[str]] = None,
    ) -> "IndependentPHC":
        """Learn a DAG independently per context, then compare.

        Parameters
        ----------
        datasets : Dict[str, NDArray]
            ``{context_label: (n_samples, n_vars)}`` arrays.
        context_labels : list of str, optional
            Explicit ordering of context labels; defaults to sorted keys.

        Returns
        -------
        self
        """
        if not datasets:
            raise ValueError("datasets must be non-empty")
        if isinstance(datasets, list):
            datasets = {f"ctx_{i}": d for i, d in enumerate(datasets)}
        first = next(iter(datasets.values()))
        self._n_vars = first.shape[1]
        for key, data in datasets.items():
            if data.shape[1] != self._n_vars:
                raise ValueError(
                    f"Context {key!r} has {data.shape[1]} variables, "
                    f"expected {self._n_vars}"
                )

        self._dags = self._learn_per_context(datasets)
        self._results = self._post_hoc_comparison(self._dags)
        self._plasticity = self._classify_all_edges(self._dags)
        self._fitted = True
        return self

    def predict_plasticity(self) -> Dict[Tuple[int, int], PlasticityClass]:
        """Return plasticity classification for every edge.

        Returns
        -------
        Dict[Tuple[int, int], PlasticityClass]
            ``{(i, j): PlasticityClass}`` for each variable pair with
            an edge in at least one context.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        return dict(self._plasticity)

    def per_context_dags(self) -> Dict[str, NDArray]:
        """Return the CPDAG adjacency matrices learned per context."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        return dict(self._dags)

    def compare(self) -> Dict[str, Any]:
        """Return pairwise comparison results."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        return dict(self._results)

    def aggregate_results(self) -> NDArray:
        """Return pairwise SHD matrix of shape ``(n_ctx, n_ctx)``."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        keys = sorted(self._dags.keys())
        n = len(keys)
        mat = np.zeros((n, n), dtype=np.float64)
        for a in range(n):
            for b in range(a + 1, n):
                shd = _structural_hamming_distance(
                    self._dags[keys[a]], self._dags[keys[b]],
                )
                mat[a, b] = mat[b, a] = float(shd)
        return mat

    # ---------------------------------------------------------------
    # Internal methods
    # ---------------------------------------------------------------

    def _learn_per_context(
        self, datasets: Dict[str, NDArray],
    ) -> Dict[str, NDArray]:
        """Run structure learning independently in each context."""
        dags: Dict[str, NDArray] = {}
        for ctx_label, data in datasets.items():
            dags[ctx_label] = _pc_algorithm(data, self._alpha)
        return dags

    def _post_hoc_comparison(
        self, dags: Dict[str, NDArray],
    ) -> Dict[str, Any]:
        """Compare DAGs pairwise using the chosen metric."""
        keys = sorted(dags.keys())
        results: Dict[str, Any] = {}
        for a, b in itertools.combinations(keys, 2):
            pair_key = f"{a}__vs__{b}"
            if self._comparison_method == "shd":
                shd = _structural_hamming_distance(dags[a], dags[b])
                results[pair_key] = {
                    "shd": shd,
                    "edges_a": int(np.sum(dags[a] != 0)),
                    "edges_b": int(np.sum(dags[b] != 0)),
                }
            else:
                edges_a = _collect_edges(dags[a])
                edges_b = _collect_edges(dags[b])
                results[pair_key] = {
                    "shared": len(edges_a & edges_b),
                    "only_a": len(edges_a - edges_b),
                    "only_b": len(edges_b - edges_a),
                    "jaccard": (
                        len(edges_a & edges_b) / max(len(edges_a | edges_b), 1)
                    ),
                }
        return results

    def _classify_edge(
        self,
        i: int,
        j: int,
        dags: Dict[str, NDArray],
    ) -> PlasticityClass:
        """Classify a single edge (i, j) by presence/absence across contexts.

        Rules
        -----
        - Present in *all* contexts → INVARIANT
        - Present in *no* contexts → skip (caller should not call)
        - Present in some but not all → STRUCTURAL_PLASTIC
        - Present in all but edge is in both directions in some contexts
          (i.e., orientation differs) → PARAMETRIC_PLASTIC as a proxy
        - Present in exactly one context → EMERGENT
        """
        ctx_keys = sorted(dags.keys())
        n_ctx = len(ctx_keys)

        present = [dags[k][i, j] != 0 for k in ctx_keys]
        reverse = [dags[k][j, i] != 0 for k in ctx_keys]
        n_present = sum(present)

        if n_present == 0:
            return PlasticityClass.INVARIANT  # no edge at all – invariant

        if n_present == n_ctx:
            # Edge present everywhere; check if orientation is consistent
            orientations_same = all(
                (present[c] == present[0] and reverse[c] == reverse[0])
                for c in range(n_ctx)
            )
            if orientations_same:
                return PlasticityClass.INVARIANT
            return PlasticityClass.PARAMETRIC_PLASTIC

        if n_present == 1:
            return PlasticityClass.EMERGENT

        return PlasticityClass.STRUCTURAL_PLASTIC

    def _classify_all_edges(
        self, dags: Dict[str, NDArray],
    ) -> Dict[Tuple[int, int], PlasticityClass]:
        """Classify plasticity for all edges that appear in any context."""
        p = self._n_vars
        all_edges: Set[Tuple[int, int]] = set()
        for dag in dags.values():
            all_edges |= _collect_edges(dag)

        classifications: Dict[Tuple[int, int], PlasticityClass] = {}
        for i, j in all_edges:
            if (i, j) in classifications or (j, i) in classifications:
                continue
            cls = self._classify_edge(i, j, dags)
            classifications[(i, j)] = cls
        return classifications

    def _structural_hamming_distance(
        self, dag1: NDArray, dag2: NDArray,
    ) -> int:
        """SHD between two DAGs (instance-method wrapper)."""
        return _structural_hamming_distance(dag1, dag2)
