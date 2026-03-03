"""CD-NOD baseline (BL4) – causal discovery from nonstationary/heterogeneous data.

Implements the CD-NOD method of Zhang et al. (2017).  CD-NOD detects
changing causal modules by augmenting the variable set with a context
indicator variable C and applying constraint-based learning.  Edges
involving C reveal which mechanisms vary across contexts.

Key algorithm
-------------
1. Augment data with a context variable C (one-hot or ordinal).
2. Run constraint-based discovery (PC algorithm) on the augmented data.
3. Edges from C to a target variable X_j indicate that the mechanism
   generating X_j changes across contexts.
4. Classify edges: those whose target has a C-edge are plastic; others
   are invariant.

References
----------
Zhang, K., Huang, B., Zhang, J., Glymour, C. & Schölkopf, B. (2017).
Causal Discovery from Nonstationary/Heterogeneous Data: Skeleton
Estimation and Orientation Determination.  *IJCAI*.
"""

from __future__ import annotations

import itertools
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sp_stats

from cpa.core.types import PlasticityClass
from cpa.baselines.ind_phc import (
    _pc_algorithm,
    _pc_skeleton,
    _orient_v_structures,
    _apply_meek_rules,
    _fisher_z_test,
    _partial_correlation,
    _collect_edges,
    _structural_hamming_distance,
)


# -------------------------------------------------------------------
# Augmentation helpers
# -------------------------------------------------------------------


def _build_context_indicator(
    datasets: Dict[str, NDArray],
    encoding: str = "ordinal",
) -> Tuple[NDArray, NDArray, List[str]]:
    """Augment datasets with a context indicator variable.

    Parameters
    ----------
    datasets : Dict[str, NDArray]
        ``{context_label: (n_i, p)}`` arrays.
    encoding : str
        ``"ordinal"`` (single column) or ``"onehot"`` (K columns).

    Returns
    -------
    augmented : NDArray, shape (N, p + n_context_cols)
    context_col_indices : NDArray, column indices of context variables
    context_keys : list of str, ordered context labels
    """
    ctx_keys = sorted(datasets.keys())
    n_ctx = len(ctx_keys)
    first = datasets[ctx_keys[0]]
    p = first.shape[1]

    if encoding == "ordinal":
        blocks = []
        for k, key in enumerate(ctx_keys):
            data = datasets[key]
            n_i = data.shape[0]
            c_col = np.full((n_i, 1), float(k))
            blocks.append(np.hstack([data, c_col]))
        augmented = np.vstack(blocks)
        context_col_indices = np.array([p], dtype=int)
    else:  # onehot
        blocks = []
        for k, key in enumerate(ctx_keys):
            data = datasets[key]
            n_i = data.shape[0]
            onehot = np.zeros((n_i, n_ctx), dtype=np.float64)
            onehot[:, k] = 1.0
            blocks.append(np.hstack([data, onehot]))
        augmented = np.vstack(blocks)
        context_col_indices = np.arange(p, p + n_ctx, dtype=int)

    return augmented, context_col_indices, ctx_keys


def _kernel_ci_test(
    data: NDArray, i: int, j: int, cond: List[int], alpha: float,
) -> Tuple[bool, float]:
    """Kernel-based conditional independence test (HSIC approximation).

    Falls back to Fisher-z for speed; uses a nonlinear residual check
    when conditioning set is small.
    """
    n = data.shape[0]
    if n < 10:
        return True, 1.0

    # Use Fisher-z as the primary test for continuous variables
    indep, p_val = _fisher_z_test(data, i, j, cond, alpha)

    # For the context indicator (which may be discrete), also check
    # correlation ratio if the variable takes few unique values
    x_i = data[:, i]
    n_unique_i = len(np.unique(x_i))
    if n_unique_i <= 10 and len(cond) == 0:
        # ANOVA-based test: does the mean of j differ across levels of i?
        groups = []
        for val in np.unique(x_i):
            mask = x_i == val
            if np.sum(mask) > 1:
                groups.append(data[mask, j])
        if len(groups) >= 2:
            stat, p_anova = sp_stats.f_oneway(*groups)
            p_val = min(p_val, float(p_anova))
            indep = p_val > alpha

    return indep, p_val


# -------------------------------------------------------------------
# CD-NOD algorithm
# -------------------------------------------------------------------


def _cdnod_skeleton(
    data: NDArray,
    n_system: int,
    alpha: float,
    max_cond: Optional[int] = None,
) -> Tuple[NDArray, Dict[Tuple[int, int], List[int]]]:
    """Learn skeleton allowing context variables to connect to system vars.

    Context variables (indices >= n_system) can be parents but not
    children of system variables (enforced after skeleton).
    """
    p_total = data.shape[1]
    if max_cond is None:
        max_cond = p_total - 2

    adj = np.ones((p_total, p_total), dtype=np.float64) - np.eye(p_total)
    sep_sets: Dict[Tuple[int, int], List[int]] = {}

    # No edges between context variables themselves
    for i in range(n_system, p_total):
        for j in range(n_system, p_total):
            if i != j:
                adj[i, j] = 0.0

    for depth in range(max_cond + 1):
        for i in range(p_total):
            for j in range(i + 1, p_total):
                if adj[i, j] == 0:
                    continue
                neighbours = [
                    k for k in range(p_total)
                    if k != i and k != j and adj[i, k] != 0
                ]
                if len(neighbours) < depth:
                    continue
                found = False
                for cond in itertools.combinations(neighbours, depth):
                    cond_list = list(cond)
                    indep, _ = _kernel_ci_test(
                        data, i, j, cond_list, alpha,
                    )
                    if indep:
                        adj[i, j] = adj[j, i] = 0.0
                        sep_sets[(i, j)] = cond_list
                        sep_sets[(j, i)] = cond_list
                        found = True
                        break
                if found:
                    continue
    return adj, sep_sets


def _orient_cdnod(
    adj: NDArray,
    sep_sets: Dict[Tuple[int, int], List[int]],
    n_system: int,
) -> NDArray:
    """Orient edges in the CD-NOD graph.

    Context variables can only be parents (C -> X_j), never children.
    Then apply v-structure and Meek rules.
    """
    p_total = adj.shape[0]
    dag = adj.copy()

    # Force orientation: context -> system (never system -> context)
    for c in range(n_system, p_total):
        for j in range(n_system):
            if dag[c, j] != 0 and dag[j, c] != 0:
                dag[j, c] = 0.0  # orient as c -> j

    # Orient v-structures among system variables
    dag = _orient_v_structures(dag, sep_sets)

    # Apply Meek rules
    dag = _apply_meek_rules(dag)

    return dag


# -------------------------------------------------------------------
# Main class
# -------------------------------------------------------------------


class CDNODBaseline:
    """CD-NOD baseline for heterogeneous/nonstationary data (BL4).

    Augments the variable set with context indicators, runs constraint-based
    discovery, and classifies edges based on context-variable adjacency.

    Parameters
    ----------
    significance_level : float
        Alpha for conditional independence tests.
    independence_test : str
        Test type (``"fisher_z"`` or ``"kernel"``).
    max_conditioning_set : int or None
        Maximum conditioning-set size.
    context_encoding : str
        How to encode contexts: ``"ordinal"`` or ``"onehot"``.
    """

    def __init__(
        self,
        significance_level: float = 0.05,
        independence_test: str = "fisher_z",
        max_conditioning_set: Optional[int] = None,
        context_encoding: str = "ordinal",
    ) -> None:
        self._alpha = significance_level
        self._test_type = independence_test
        self._max_cond = max_conditioning_set
        self._encoding = context_encoding
        self._graph: Optional[NDArray] = None
        self._system_graph: Optional[NDArray] = None
        self._context_col_indices: Optional[NDArray] = None
        self._changing: Dict[int, List[str]] = {}
        self._context_keys: List[str] = []
        self._n_vars: int = 0
        self._plasticity: Dict[Tuple[int, int], PlasticityClass] = {}
        self._datasets: Dict[str, NDArray] = {}
        self._fitted: bool = False

    # ---------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------

    def fit(
        self,
        datasets: Dict[str, NDArray],
        context_labels: Optional[List[str]] = None,
        context_variables: Optional[NDArray] = None,
    ) -> "CDNODBaseline":
        """Fit CD-NOD on multi-context data.

        Parameters
        ----------
        datasets : Dict[str, NDArray]
            ``{context_label: (n_samples, n_vars)}`` arrays.
        context_labels : list of str, optional
            Explicit ordering (default: sorted keys).
        context_variables : NDArray, optional
            Pre-built context indicator matrix (overrides auto-construction).

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
        self._datasets = dict(datasets)

        for k, d in datasets.items():
            if d.shape[1] != self._n_vars:
                raise ValueError(
                    f"Context {k!r}: {d.shape[1]} vars, expected {self._n_vars}"
                )

        # Augment with context variable(s)
        aug_data, ctx_cols, ctx_keys = self._augment_with_context(
            datasets, context_labels,
        )
        self._context_col_indices = ctx_cols
        self._context_keys = ctx_keys

        # Run constraint-based discovery on augmented data
        full_dag = self._constraint_based_discovery(aug_data)
        self._graph = full_dag

        # Extract system-variable subgraph
        self._system_graph = full_dag[:self._n_vars, :self._n_vars].copy()

        # Identify changing modules
        self._changing = self._identify_changing_modules(full_dag)

        # Classify edges
        self._plasticity = self._separate_invariant_plastic(
            self._system_graph, self._changing,
        )

        self._fitted = True
        return self

    def predict_plasticity(self) -> Dict[Tuple[int, int], PlasticityClass]:
        """Return edge plasticity classifications."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        return dict(self._plasticity)

    def learned_graph(self) -> NDArray:
        """Return the full augmented causal graph."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        assert self._graph is not None
        return self._graph.copy()

    def system_graph(self) -> NDArray:
        """Return the system-variable subgraph."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        assert self._system_graph is not None
        return self._system_graph.copy()

    def changing_modules(self) -> Dict[int, List[str]]:
        """Return variables whose mechanisms change across contexts."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        return dict(self._changing)

    def context_specific_edges(self) -> Dict[str, NDArray]:
        """Return per-context active edges via per-context PC.

        For each context, runs PC independently and returns the DAG.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        results: Dict[str, NDArray] = {}
        for key, data in self._datasets.items():
            results[key] = _pc_algorithm(data, self._alpha)
        return results

    # ---------------------------------------------------------------
    # Internal methods
    # ---------------------------------------------------------------

    def _augment_with_context(
        self,
        datasets: Dict[str, NDArray],
        context_labels: Optional[List[str]] = None,
    ) -> Tuple[NDArray, NDArray, List[str]]:
        """Add context variable C to the data."""
        return _build_context_indicator(datasets, self._encoding)

    def _constraint_based_discovery(self, augmented_data: NDArray) -> NDArray:
        """Run PC-like discovery on the augmented data."""
        skeleton, sep_sets = _cdnod_skeleton(
            augmented_data, self._n_vars, self._alpha, self._max_cond,
        )
        dag = _orient_cdnod(skeleton, sep_sets, self._n_vars)
        return dag

    def _identify_changing_modules(
        self, dag: NDArray,
    ) -> Dict[int, List[str]]:
        """Identify system variables adjacent to context variable(s).

        If C -> X_j exists, the mechanism for X_j changes across contexts.
        """
        assert self._context_col_indices is not None
        changing: Dict[int, List[str]] = {}

        for c_idx in self._context_col_indices:
            for j in range(self._n_vars):
                # Check if context variable has an edge to j
                if dag[c_idx, j] != 0:
                    if j not in changing:
                        changing[j] = list(self._context_keys)
                    # All contexts are affected (the mechanism changes)

        return changing

    def _separate_invariant_plastic(
        self,
        system_dag: NDArray,
        changing: Dict[int, List[str]],
    ) -> Dict[Tuple[int, int], PlasticityClass]:
        """Classify edges based on whether their target has a changing module.

        Rules:
        - Edge (i, j) where j has a changing module → PARAMETRIC_PLASTIC
          (structure is detected, but mechanism varies)
        - Edge (i, j) where j does NOT have a changing module → INVARIANT
        - If the edge itself is absent in per-context runs for some
          contexts, escalate to STRUCTURAL_PLASTIC
        """
        edges = _collect_edges(system_dag)
        classifications: Dict[Tuple[int, int], PlasticityClass] = {}

        # Get per-context DAGs for finer classification
        per_ctx_dags: Dict[str, NDArray] = {}
        for key, data in self._datasets.items():
            per_ctx_dags[key] = _pc_algorithm(data, self._alpha)

        n_ctx = len(self._context_keys)

        for i, j in edges:
            if (i, j) in classifications or (j, i) in classifications:
                continue

            # Count edge presence across per-context DAGs
            n_present = sum(
                1 for key in self._context_keys
                if per_ctx_dags[key][i, j] != 0
            )

            if j in changing:
                if n_present < n_ctx and n_present > 0:
                    classifications[(i, j)] = PlasticityClass.STRUCTURAL_PLASTIC
                elif n_present == 1:
                    classifications[(i, j)] = PlasticityClass.EMERGENT
                else:
                    classifications[(i, j)] = PlasticityClass.PARAMETRIC_PLASTIC
            else:
                if n_present == n_ctx or n_present == 0:
                    classifications[(i, j)] = PlasticityClass.INVARIANT
                elif n_present == 1:
                    classifications[(i, j)] = PlasticityClass.EMERGENT
                else:
                    classifications[(i, j)] = PlasticityClass.STRUCTURAL_PLASTIC

        return classifications
