"""Joint Causal Inference (JCI) baseline (BL5).

Implements the JCI framework of Mooij et al. (2020).  JCI models
context indicators as explicit "system variables" in an augmented
causal graph, with the constraint that target variables cannot cause
system variables.

Key idea
--------
1. Construct system data by adding one-hot context indicators as new
   variables (C_1, …, C_K) to the observation matrix.
2. Impose background-knowledge constraints:
   - No edges from target variables to system variables.
   - (JCI-1) System variables are exogenous.
   - (JCI-123) System variables may cause target variables directly.
3. Run constrained structure learning.
4. Extract the target-variable subgraph and classify edges.

References
----------
Mooij, Magliacane, & Claassen (2020).  Joint Causal Inference from
Multiple Contexts.  *JMLR*, 21(99), 1-108.
"""

from __future__ import annotations

import itertools
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sp_stats

from cpa.core.types import PlasticityClass
from cpa.baselines.ind_phc import (
    _pc_skeleton,
    _orient_v_structures,
    _apply_meek_rules,
    _collect_edges,
    _fisher_z_test,
)


# -------------------------------------------------------------------
# System data construction
# -------------------------------------------------------------------


def _construct_system_data(
    datasets: Dict[str, NDArray],
    context_keys: List[str],
) -> Tuple[NDArray, int, int]:
    """Augment data with context indicator (system) variables.

    For K contexts, adds K binary columns.  Each row has exactly one
    context indicator set to 1.

    Parameters
    ----------
    datasets : Dict[str, NDArray]
    context_keys : ordered context labels

    Returns
    -------
    augmented : NDArray, shape (N, p + K)
    num_system : int, number of system variables (K)
    num_target : int, number of target variables (p)
    """
    K = len(context_keys)
    first = datasets[context_keys[0]]
    p = first.shape[1]

    blocks = []
    for k, key in enumerate(context_keys):
        data = datasets[key]
        n_i = data.shape[0]
        indicators = np.zeros((n_i, K), dtype=np.float64)
        indicators[:, k] = 1.0
        # System variables first, then target variables
        blocks.append(np.hstack([indicators, data]))
    augmented = np.vstack(blocks)
    return augmented, K, p


# -------------------------------------------------------------------
# JCI constraint generation
# -------------------------------------------------------------------


def _jci_constraints(
    num_system: int, num_target: int, jci_type: str = "jci-1",
) -> Dict[str, Any]:
    """Generate JCI background-knowledge constraints.

    Parameters
    ----------
    num_system : int (K)
    num_target : int (p)
    jci_type : str
        ``"jci-1"``: system vars are exogenous, no target -> system edges.
        ``"jci-123"``: full JCI with direct system -> target edges allowed.

    Returns
    -------
    constraints : dict with keys:
        "forbidden" : list of (i, j) edges that are forbidden
        "required"  : list of (i, j) edges that are required
    """
    total = num_system + num_target
    forbidden: List[Tuple[int, int]] = []
    required: List[Tuple[int, int]] = []

    # Core JCI constraint: no edges from target -> system
    for t in range(num_system, total):
        for s in range(num_system):
            forbidden.append((t, s))

    if jci_type == "jci-1":
        # System variables are exogenous: no edges between system vars
        for s1 in range(num_system):
            for s2 in range(num_system):
                if s1 != s2:
                    forbidden.append((s1, s2))

    return {"forbidden": forbidden, "required": required}


# -------------------------------------------------------------------
# Constrained PC algorithm
# -------------------------------------------------------------------


def _constrained_pc_skeleton(
    data: NDArray,
    alpha: float,
    constraints: Dict[str, Any],
    max_cond: Optional[int] = None,
) -> Tuple[NDArray, Dict[Tuple[int, int], List[int]]]:
    """PC skeleton search with forbidden/required edge constraints."""
    p = data.shape[1]
    adj = np.ones((p, p), dtype=np.float64) - np.eye(p)
    sep_sets: Dict[Tuple[int, int], List[int]] = {}

    # Apply forbidden edges
    forbidden = set()
    for i, j in constraints.get("forbidden", []):
        forbidden.add((i, j))
        adj[i, j] = 0.0

    # Apply required edges (prevent removal)
    required = set()
    for i, j in constraints.get("required", []):
        required.add((i, j))
        required.add((j, i))

    if max_cond is None:
        max_cond = p - 2

    for depth in range(max_cond + 1):
        for i in range(p):
            for j in range(i + 1, p):
                if adj[i, j] == 0:
                    continue
                if (i, j) in required or (j, i) in required:
                    continue
                neighbours = [
                    k for k in range(p) if k != i and k != j and adj[i, k] != 0
                ]
                if len(neighbours) < depth:
                    continue
                found = False
                for cond in itertools.combinations(neighbours, depth):
                    cond_list = list(cond)
                    indep, _ = _fisher_z_test(data, i, j, cond_list, alpha)
                    if indep:
                        # Don't remove forbidden-direction edges that are
                        # already zero; remove both directions of the
                        # undirected skeleton edge
                        if (i, j) not in forbidden:
                            adj[i, j] = 0.0
                        if (j, i) not in forbidden:
                            adj[j, i] = 0.0
                        sep_sets[(i, j)] = cond_list
                        sep_sets[(j, i)] = cond_list
                        found = True
                        break
                if found:
                    continue

    return adj, sep_sets


def _constrained_orient(
    adj: NDArray,
    sep_sets: Dict[Tuple[int, int], List[int]],
    constraints: Dict[str, Any],
    num_system: int,
) -> NDArray:
    """Orient edges with JCI constraints.

    1. Enforce forbidden edges (remove them).
    2. Orient system -> target (system vars can only be parents).
    3. Apply v-structure rules.
    4. Apply Meek rules.
    """
    p = adj.shape[0]
    dag = adj.copy()

    # Enforce forbidden
    for i, j in constraints.get("forbidden", []):
        dag[i, j] = 0.0

    # System variables can only be parents of target variables
    for s in range(num_system):
        for t in range(num_system, p):
            if dag[s, t] != 0 and dag[t, s] != 0:
                dag[t, s] = 0.0  # orient s -> t

    # V-structures and Meek
    dag = _orient_v_structures(dag, sep_sets)

    # Re-enforce forbidden after v-structure orientation
    for i, j in constraints.get("forbidden", []):
        dag[i, j] = 0.0

    dag = _apply_meek_rules(dag)

    # Final enforcement
    for i, j in constraints.get("forbidden", []):
        dag[i, j] = 0.0

    return dag


# -------------------------------------------------------------------
# Main class
# -------------------------------------------------------------------


class JCIBaseline:
    """Joint Causal Inference baseline (BL5).

    Models context indicators as system variables with structural
    constraints and learns the joint causal graph.

    Parameters
    ----------
    significance_level : float
        Alpha for conditional independence tests.
    jci_type : str
        JCI variant: ``"jci-1"`` (exogenous system vars) or
        ``"jci-123"`` (full JCI).
    method : str
        Underlying method (``"pc"`` or ``"fci"``).
    """

    def __init__(
        self,
        significance_level: float = 0.05,
        jci_type: str = "jci-1",
        method: str = "pc",
    ) -> None:
        self._alpha = significance_level
        self._jci_type = jci_type
        self._method = method
        self._graph: Optional[NDArray] = None
        self._system_graph: Optional[NDArray] = None
        self._target_graph: Optional[NDArray] = None
        self._num_system: int = 0
        self._num_target: int = 0
        self._context_keys: List[str] = []
        self._datasets: Dict[str, NDArray] = {}
        self._plasticity: Dict[Tuple[int, int], PlasticityClass] = {}
        self._fitted: bool = False

    # ---------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------

    def fit(
        self,
        datasets: Dict[str, NDArray],
        context_labels: Optional[List[str]] = None,
        intervention_targets: Optional[Dict[str, List[int]]] = None,
    ) -> "JCIBaseline":
        """Fit JCI on multi-context data.

        Parameters
        ----------
        datasets : Dict[str, NDArray]
            ``{context_label: (n_samples, n_vars)}`` arrays.
        context_labels : list of str, optional
        intervention_targets : Dict[str, List[int]], optional

        Returns
        -------
        self
        """
        if not datasets:
            raise ValueError("datasets must be non-empty")
        if isinstance(datasets, list):
            datasets = {f"ctx_{i}": d for i, d in enumerate(datasets)}

        self._datasets = dict(datasets)
        self._context_keys = sorted(datasets.keys())
        first = datasets[self._context_keys[0]]
        self._num_target = first.shape[1]

        for k, d in datasets.items():
            if d.shape[1] != self._num_target:
                raise ValueError(
                    f"Context {k!r}: {d.shape[1]} vars, "
                    f"expected {self._num_target}"
                )

        # 1. Construct augmented data
        augmented, num_sys, num_tgt = _construct_system_data(
            datasets, self._context_keys,
        )
        self._num_system = num_sys

        # 2. Generate JCI constraints
        constraints = _jci_constraints(num_sys, num_tgt, self._jci_type)

        # 3. Constrained structure learning
        full_dag = self._structure_learning_with_constraints(
            augmented, constraints,
        )
        self._graph = full_dag

        # 4. Extract subgraphs
        total = num_sys + num_tgt
        self._system_graph = full_dag[:num_sys, :num_sys].copy()
        self._target_graph = full_dag[num_sys:, num_sys:].copy()

        # 5. Classify edges
        self._plasticity = self._classify_edges(full_dag)

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

    def system_variables_graph(self) -> NDArray:
        """Return the graph restricted to system variables only."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        assert self._system_graph is not None
        return self._system_graph.copy()

    def target_variables_graph(self) -> NDArray:
        """Return the graph restricted to target variables only."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        assert self._target_graph is not None
        return self._target_graph.copy()

    def intervention_effects(self) -> Dict[str, NDArray]:
        """Return estimated intervention effects per context.

        For each context, returns a vector indicating which target
        variables are directly affected by the context indicator.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        assert self._graph is not None
        effects: Dict[str, NDArray] = {}
        for k, key in enumerate(self._context_keys):
            # Row k in the full graph: edges from system var k to targets
            effect_vec = self._graph[k, self._num_system:].copy()
            effects[key] = effect_vec
        return effects

    def _extract_target_edges(
        self, dag: NDArray, num_system: int,
    ) -> NDArray:
        """Extract the target-variable subgraph from the full DAG."""
        return dag[num_system:, num_system:].copy()

    # ---------------------------------------------------------------
    # Internal methods
    # ---------------------------------------------------------------

    def _structure_learning_with_constraints(
        self,
        data: NDArray,
        constraints: Dict[str, Any],
    ) -> NDArray:
        """Run constrained PC on the augmented data."""
        skeleton, sep_sets = _constrained_pc_skeleton(
            data, self._alpha, constraints,
        )
        dag = _constrained_orient(
            skeleton, sep_sets, constraints, self._num_system,
        )
        return dag

    def _classify_edges(
        self, full_dag: NDArray,
    ) -> Dict[Tuple[int, int], PlasticityClass]:
        """Classify target-variable edges based on system-variable adjacency.

        Rules:
        - Edge (i, j) in target graph where some system var C_k -> j:
          the mechanism for j changes across contexts → PARAMETRIC_PLASTIC
        - Edge (i, j) where no system var -> j: → INVARIANT
        - Determine STRUCTURAL vs EMERGENT by checking per-context presence
        """
        target_dag = self._target_graph
        assert target_dag is not None
        assert self._graph is not None

        edges = _collect_edges(target_dag)
        classifications: Dict[Tuple[int, int], PlasticityClass] = {}

        # Which target variables are affected by system variables?
        affected_targets: Set[int] = set()
        for s in range(self._num_system):
            for t in range(self._num_target):
                if self._graph[s, self._num_system + t] != 0:
                    affected_targets.add(t)

        # Per-context DAGs for finer classification
        per_ctx_dags: Dict[str, NDArray] = {}
        for key, data in self._datasets.items():
            from cpa.baselines.ind_phc import _pc_algorithm
            per_ctx_dags[key] = _pc_algorithm(data, self._alpha)

        n_ctx = len(self._context_keys)

        for i, j in edges:
            if (i, j) in classifications or (j, i) in classifications:
                continue

            n_present = sum(
                1 for key in self._context_keys
                if per_ctx_dags[key][i, j] != 0
            )

            if j in affected_targets:
                if n_present < n_ctx and n_present > 1:
                    classifications[(i, j)] = PlasticityClass.STRUCTURAL_PLASTIC
                elif n_present == 1:
                    classifications[(i, j)] = PlasticityClass.EMERGENT
                else:
                    classifications[(i, j)] = PlasticityClass.PARAMETRIC_PLASTIC
            else:
                if n_present >= n_ctx:
                    classifications[(i, j)] = PlasticityClass.INVARIANT
                elif n_present == 1:
                    classifications[(i, j)] = PlasticityClass.EMERGENT
                else:
                    classifications[(i, j)] = PlasticityClass.STRUCTURAL_PLASTIC

        return classifications
